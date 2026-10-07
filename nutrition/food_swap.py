"""
CliNexa Healthcare Intelligence Platform
Module: Smart Food Swap Engine
Description: Evidence-based healthier food substitutions with nutritional explanations,
calorie impact, and physiological rationales.
"""

from typing import List, Dict, Any, Optional

FOOD_SWAPS_CATALOG: List[Dict[str, Any]] = [
    {
        "category": "Beverages",
        "original": "Regular Sugary Soda (330ml)",
        "alternative": "Sparkling Water with Fresh Mint & Lime",
        "calories_original": 140,
        "calories_swap": 5,
        "calorie_savings": 135,
        "macros_benefit": "Saves ~35g of refined high-fructose corn syrup / sucrose.",
        "scientific_rationale": "Eliminates rapid glycemic spikes, reduces hepatic fat accumulation burden, and maintains optimal hydration without artificial sweeteners."
    },
    {
        "category": "Beverages",
        "original": "Commercial Sweetened Fruit Juice (250ml)",
        "alternative": "Whole Orange or Fresh Berries Infused Water",
        "calories_original": 125,
        "calories_swap": 45,
        "calorie_savings": 80,
        "macros_benefit": "Restores 3.5g of natural dietary pectin and soluble fiber.",
        "scientific_rationale": "Juicing removes vital fruit matrix fiber, leading to swift insulin surges. Whole fruit preserves prebiotic fibers that slow gastric emptying."
    },
    {
        "category": "Beverages",
        "original": "Caramel Flavored Latte with Whole Milk",
        "alternative": "Americano or Espresso with Steamed Unsweetened Oat Milk & Cinnamon",
        "calories_original": 280,
        "calories_swap": 55,
        "calorie_savings": 225,
        "macros_benefit": "Reduces saturated fat by 7g and cuts refined sugar by 28g.",
        "scientific_rationale": "Cinnamon provides natural aromatic sweetness and polyphenols shown to support insulin sensitivity, without glycemic burden."
    },
    {
        "category": "Snacks",
        "original": "Commercial Potato Chips (50g bag)",
        "alternative": "Air-Popped Spiced Popcorn or Roasted Chickpeas",
        "calories_original": 270,
        "calories_swap": 115,
        "calorie_savings": 155,
        "macros_benefit": "Replaces oxidized frying oils with 5g plant protein and 4g prebiotic fiber.",
        "scientific_rationale": "Deep-fried starches generate acrylamides and high advanced glycation end-products. Whole legume snacks increase satiety peptides like GLP-1 and PYY."
    },
    {
        "category": "Snacks",
        "original": "Milk Chocolate Candy Bar (50g)",
        "alternative": "Dark Chocolate (85% Cocoa) with Raw Walnuts (25g)",
        "calories_original": 260,
        "calories_swap": 160,
        "calorie_savings": 100,
        "macros_benefit": "Rich in flavanols (epicatechin) and plant-based ALA Omega-3 fatty acids.",
        "scientific_rationale": "High-cocoa flavonoids stimulate endothelial nitric oxide synthase, enhancing arterial compliance and reducing systemic oxidative stress."
    },
    {
        "category": "Snacks",
        "original": "Store-Bought Frosted Donut",
        "alternative": "Baked Apple Slices Dusted with Ceylon Cinnamon & Crushed Almonds",
        "calories_original": 310,
        "calories_swap": 120,
        "calorie_savings": 190,
        "macros_benefit": "Cuts trans-fats and 22g refined sugar; adds 4g soluble fiber.",
        "scientific_rationale": "Pectin fiber forms a viscous gel in the intestinal tract that attenuates postprandial glucose excursions and lowers LDL cholesterol reabsorption."
    },
    {
        "category": "Grains & Carbs",
        "original": "Refined White Bread (2 slices)",
        "alternative": "100% Sprouted Whole Grain or Sourdough Rye Bread",
        "calories_original": 160,
        "calories_swap": 140,
        "calorie_savings": 20,
        "macros_benefit": "Triples dietary fiber (5g vs 1.5g) and increases bioavailable minerals.",
        "scientific_rationale": "Sprouting and long fermentation partially break down phytates and resistant starches, lowering glycemic response and enhancing micronutrient absorption."
    },
    {
        "category": "Grains & Carbs",
        "original": "Instant White Rice (1 cup cooked)",
        "alternative": "Quinoa or Steamed Brown Jasmine Rice (1 cup)",
        "calories_original": 210,
        "calories_swap": 180,
        "calorie_savings": 30,
        "macros_benefit": "Provides all 9 essential amino acids plus 5g fiber and magnesium.",
        "scientific_rationale": "The intact bran and germ layers contain magnesium and B vitamins essential for mitochondrial glucose metabolism and sustained energy release."
    },
    {
        "category": "Grains & Carbs",
        "original": "Traditional Durum Wheat Pasta (1 cup cooked)",
        "alternative": "Zucchini Ribbons (Zoodles) or Red Lentil Pasta",
        "calories_original": 220,
        "calories_swap": 45,
        "calorie_savings": 175,
        "macros_benefit": "Cuts net carbohydrates by 38g while supplying bioflavonoids and potassium.",
        "scientific_rationale": "Substituting vegetable bases reduces caloric density substantially while maintaining meal volume and promoting healthy gut motility."
    },
    {
        "category": "Proteins & Mains",
        "original": "Deep-Fried Battered Chicken Nuggets",
        "alternative": "Oven-Baked Herb-Crusted Chicken Tenderloins or Crispy Tofu",
        "calories_original": 340,
        "calories_swap": 170,
        "calorie_savings": 170,
        "macros_benefit": "Cuts 18g inflammatory seed oil fat while preserving 26g high-quality protein.",
        "scientific_rationale": "Minimizes lipid peroxidation products and trans fats associated with repeated frying oils, supporting cardiovascular endothelial integrity."
    },
    {
        "category": "Proteins & Mains",
        "original": "Processed Hot Dog or High-Sodium Sausage",
        "alternative": "Spiced Black Bean Patty or Grilled Turkey/Tofu Sausage",
        "calories_original": 290,
        "calories_swap": 145,
        "calorie_savings": 145,
        "macros_benefit": "Lowers sodium by 600mg and reduces saturated fat by 9g.",
        "scientific_rationale": "Eliminates carcinogenic sodium nitrites and nitrates commonly found in cured meats, mitigating colorectal health risks."
    },
    {
        "category": "Dairy & Condiments",
        "original": "Mayonnaise (2 tbsp, 30g)",
        "alternative": "Mashed Avocado with Lemon Juice or Greek Yogurt Dressing",
        "calories_original": 190,
        "calories_swap": 50,
        "calorie_savings": 140,
        "macros_benefit": "Replaces soybean oil with heart-healthy monounsaturated oleic acid and probiotics.",
        "scientific_rationale": "Monounsaturated fats help maintain optimal HDL-to-LDL cholesterol ratios while Greek yogurt adds probiotic bacterial strains supporting gut microflora."
    },
    {
        "category": "Dairy & Condiments",
        "original": "Commercial Sweet BBQ Sauce or Ketchup (3 tbsp)",
        "alternative": "Fresh Tomato Salsa or Roasted Red Pepper Chimichurri",
        "calories_original": 90,
        "calories_swap": 20,
        "calorie_savings": 70,
        "macros_benefit": "Cuts 18g high-fructose corn syrup; provides bioavailable lycopene.",
        "scientific_rationale": "Lycopene is a potent lipid-soluble carotenoid antioxidant that protects vascular endothelial membranes from oxidative damage."
    },
    {
        "category": "Desserts",
        "original": "Traditional Full-Cream Ice Cream (1 cup)",
        "alternative": "Frozen Blended Banana 'Nice Cream' with Chia Seeds",
        "calories_original": 280,
        "calories_swap": 120,
        "calorie_savings": 160,
        "macros_benefit": "Zero added sugar, 400mg potassium, and 4g prebiotic resistant starch.",
        "scientific_rationale": "Cold bananas provide resistant starch that escapes small intestinal enzymatic digestion, nourishing butyrate-producing colonocytes in the large bowel."
    }
]


def get_all_swaps() -> List[Dict[str, Any]]:
    """Return the entire catalog of food swaps."""
    return FOOD_SWAPS_CATALOG


def get_categories() -> List[str]:
    """Return unique categories of food swaps."""
    categories = sorted(list(set(item["category"] for item in FOOD_SWAPS_CATALOG)))
    return categories


def search_swaps(query: str = "", category: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search food swaps by term and optional category filter.
    """
    results = []
    q = query.lower().strip()
    for item in FOOD_SWAPS_CATALOG:
        if category and category != "All" and item["category"] != category:
            continue
        if q:
            in_orig = q in item["original"].lower()
            in_alt = q in item["alternative"].lower()
            in_cat = q in item["category"].lower()
            in_rat = q in item["scientific_rationale"].lower()
            if not (in_orig or in_alt or in_cat or in_rat):
                continue
        results.append(item)
    return results
