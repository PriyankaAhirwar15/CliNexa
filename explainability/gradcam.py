"""
CliNexa Healthcare Intelligence Platform
Module: Grad-CAM Explainability (Gradient-weighted Class Activation Mapping)
Description: Generates visual explanations for decisions made by the ResNet-50
deep convolutional network, highlighting localized radiological focus areas.

CRITICAL REQUIREMENT:
Produces 3 visual representations:
1. Original medical image
2. Grad-CAM colormap heatmap
3. Superimposed heatmap overlay on original image
With required clinical safety explanation.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm
from models.vision.resnet_classifier import MedicalVisionService, IMAGE_TRANSFORMS


class GradCAMExplainer:
    """
    Grad-CAM engine hooking into ResNet-50 final convolutional block.
    """
    def __init__(self, vision_service: Optional[MedicalVisionService] = None):
        self.service = vision_service or MedicalVisionService()
        self.model = self.service.model
        # Hook target: the last bottleneck block of layer4
        self.target_layer = self.model.backbone.layer4[-1]

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target conv layer."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        image: Image.Image,
        target_class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute Grad-CAM 2D heatmap matrix normalized between 0 and 1.
        """
        self.model.eval()
        if image.mode != "RGB":
            image = image.convert("RGB")

        orig_w, orig_h = image.size
        input_tensor = IMAGE_TRANSFORMS(image).unsqueeze(0)
        input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(probs).item())

        confidence = float(probs[target_class_idx].item())

        # Backward pass with respect to target class logit
        self.model.zero_grad()
        score = logits[0, target_class_idx]
        score.backward()

        # Gradients: [1, 2048, 7, 7], Activations: [1, 2048, 7, 7]
        gradients = self.gradients.detach()
        activations = self.activations.detach()

        # Global average pooling of gradients over spatial dimensions (H, W)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        # Apply ReLU to retain only features having positive influence
        cam = F.relu(cam)

        # Normalize between 0 and 1
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize CAM to original image dimensions using PIL bilinear interpolation
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        resized_cam = np.array(cam_pil).astype(np.float32) / 255.0

        return resized_cam, target_class_idx, confidence

    def generate_visualizations(
        self,
        image: Image.Image,
        target_class_idx: Optional[int] = None,
        alpha: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate complete Grad-CAM explainability package containing:
        - Original PIL image
        - Colormapped heatmap PIL image
        - Blended overlay PIL image
        - Metadata & required clinical explanation
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        heatmap_norm, class_idx, conf = self.generate_heatmap(image, target_class_idx)

        # Colormap application using matplotlib 'jet'
        try:
            import matplotlib
            colormap = matplotlib.colormaps["jet"]
        except Exception:
            colormap = cm.get_cmap("jet")
        heatmap_colored = colormap(heatmap_norm)[:, :, :3]  # Drop alpha
        heatmap_uint8 = (heatmap_colored * 255).astype(np.uint8)
        heatmap_image = Image.fromarray(heatmap_uint8)

        # Overlay blend: (1 - alpha) * original + alpha * heatmap
        orig_np = np.array(image).astype(np.float32)
        blended_np = (1.0 - alpha) * orig_np + alpha * (heatmap_colored * 255.0)
        blended_uint8 = np.clip(blended_np, 0, 255).astype(np.uint8)
        overlay_image = Image.fromarray(blended_uint8)

        predicted_label = self.service.classes[class_idx]

        return {
            "original_image": image,
            "heatmap_image": heatmap_image,
            "overlay_image": overlay_image,
            "predicted_class": predicted_label,
            "confidence_percent": round(conf * 100, 1),
            "explanation_notice": (
                "Highlighted regions represent areas that contributed to the model's prediction. "
                "They should not be interpreted as a medical diagnosis."
            ),
            "methodology": "Grad-CAM (Gradient-weighted Class Activation Mapping) on ResNet-50 layer4"
        }
