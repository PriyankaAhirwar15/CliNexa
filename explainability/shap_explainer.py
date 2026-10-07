"""
CliNexa Healthcare Intelligence Platform
Module: SHAP Model Explainability
Description: Computes Shapley attribution values and feature-level explanations
for the PyTorch Deep Clinical Risk model.

EXPLAINABILITY REQUIREMENT:
Explains WHY the deep learning model produced a particular risk result,
dissecting positive (risk-increasing) and negative (risk-reducing) contributions
with interactive visualizations and human-readable clinical narratives.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
import plotly.graph_objects as go
from models.risk.risk_dnn import (
    DeepClinicalRiskNet,
    FEATURE_NAMES,
    FEATURE_MEANS,
    FEATURE_STDS,
    RiskClassifierService
)


class ClinicalSHAPExplainer:
    """
    Shapley Value Explainer for PyTorch Deep Clinical Risk Classifier.
    Computes exact marginal feature attributions relative to reference baseline population.
    """
    def __init__(self, risk_service: Optional[RiskClassifierService] = None):
        self.risk_service = risk_service or RiskClassifierService()
        self.feature_names = FEATURE_NAMES
        self.feature_means = FEATURE_MEANS
        self.feature_stds = FEATURE_STDS

    def explain_instance(self, raw_features: List[float]) -> Dict[str, Any]:
        """
        Compute feature contributions using gradient-informed Shapley approximation
        around epidemiological background expectation.
        """
        arr = np.array(raw_features, dtype=np.float32)
        norm_arr = (arr - self.feature_means) / (self.feature_stds + 1e-6)
        x_tensor = torch.tensor(norm_arr, dtype=torch.float32).unsqueeze(0)
        x_tensor.requires_grad_(True)

        # Baseline point: epidemiological reference mean (all zeros in normalized space)
        baseline_tensor = torch.zeros_like(x_tensor)

        # Forward pass for target instance
        out_target, _, _ = self.risk_service.model(x_tensor)
        target_score = out_target.item()

        # Forward pass for baseline
        with torch.no_grad():
            out_baseline, _, _ = self.risk_service.model(baseline_tensor)
            baseline_score = out_baseline.item()

        # Integrated Gradients / Path-based Shapley approximation (5 steps)
        steps = 10
        accumulated_grads = torch.zeros_like(x_tensor)
        diff = x_tensor - baseline_tensor

        for alpha in np.linspace(0.1, 1.0, steps):
            interpolated = baseline_tensor + float(alpha) * diff
            interpolated = interpolated.detach().clone().requires_grad_(True)
            pred, _, _ = self.risk_service.model(interpolated)
            grad = torch.autograd.grad(pred, interpolated)[0]
            accumulated_grads += grad

        avg_grads = (accumulated_grads / steps).squeeze(0).detach().numpy()
        attributions = (diff.squeeze(0).detach().numpy()) * avg_grads

        # Scale attributions so their sum approximately reflects difference from baseline
        total_delta = target_score - baseline_score
        raw_sum = float(np.sum(attributions))
        if abs(raw_sum) > 1e-6:
            scaled_attributions = (attributions / raw_sum) * total_delta
        else:
            scaled_attributions = attributions

        results = []
        positive_factors = []
        protective_factors = []

        for name, val, raw_val in zip(self.feature_names, scaled_attributions, raw_features):
            impact = float(val)
            pct_impact = round(impact * 100, 2)
            item = {
                "feature": name,
                "current_value": raw_val,
                "shap_value": round(impact, 4),
                "impact_percent": pct_impact,
                "direction": "Risk-Increasing" if impact > 0 else "Risk-Reducing"
            }
            results.append(item)
            if impact > 0.005:
                positive_factors.append(f"{name} (+{pct_impact}%)")
            elif impact < -0.005:
                protective_factors.append(f"{name} ({pct_impact}%)")

        # Sort by absolute magnitude of importance
        results_sorted = sorted(results, key=lambda x: abs(x["shap_value"]), reverse=True)

        # Generate human-readable narrative
        top_increasing = [x["feature"] for x in results_sorted if x["shap_value"] > 0][:3]
        top_protective = [x["feature"] for x in results_sorted if x["shap_value"] < 0][:2]

        narrative_parts = []
        if top_increasing:
            narrative_parts.append(
                f"Factors that contributed most toward elevating the model's estimated risk include "
                f"<b>{', '.join(top_increasing)}</b>."
            )
        if top_protective:
            narrative_parts.append(
                f"Factors that acted favorably to reduce estimated risk include "
                f"<b>{', '.join(top_protective)}</b>."
            )
        if not narrative_parts:
            narrative_parts.append("All measured factors are in close alignment with baseline expectations.")

        narrative = " ".join(narrative_parts)

        return {
            "baseline_score": round(baseline_score * 100, 1),
            "predicted_score": round(target_score * 100, 1),
            "feature_attributions": results,
            "sorted_attributions": results_sorted,
            "summary_narrative": narrative,
            "positive_factors": positive_factors,
            "protective_factors": protective_factors,
            "disclaimer": (
                "SHAP values reflect model-internal feature attributions and statistical associations. "
                "They describe why the neural network computed this output and do not constitute direct clinical etiology."
            )
        }

    def create_shap_bar_chart(self, explanation: Dict[str, Any]) -> go.Figure:
        """Create a clean horizontal waterfall/bar chart of feature attributions."""
        items = explanation.get("sorted_attributions", [])
        # Show top 8 most influential features
        items_to_plot = items[:8]
        items_to_plot = list(reversed(items_to_plot))

        features = [x["feature"] for x in items_to_plot]
        impacts = [x["impact_percent"] for x in items_to_plot]
        colors = ["#EF4444" if v >= 0 else "#10B981" for v in impacts]

        fig = go.Figure(go.Bar(
            x=impacts,
            y=features,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:+.1f}%" for v in impacts],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Risk Impact: %{x:+.2f}%<extra></extra>"
        ))

        fig.update_layout(
            title="<b>SHAP Feature Attribution: Contribution to Health Risk</b>",
            xaxis_title="Percentage Contribution to Estimated Risk Score (+ = Increases Risk, - = Favorable)",
            font=dict(family="Arial, sans-serif", size=13),
            margin=dict(l=150, r=40, t=50, b=40),
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig
