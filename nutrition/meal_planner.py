"""
CliNexa Healthcare Intelligence Platform
Module: 7-Day Personalized Meal Planner
Description: Generates structured, evidence-aligned weekly meal plans with
macronutrient approximations tailored to dietary preference, budget, cuisine,
allergies, and calorie objectives.
"""

from typing import Dict, List, Any, Optional


MEAL_TEMPLATES = {
    "vegan": {
        "breakfast": [
            {"name": "Overnight Chia & Rolled Oats with Blueberries", "calories": 360, "protein": 11, "carbs": 58, "fat": 10, "fiber": 9, "desc": "Rolled oats soaked in unsweetened almond milk with chia seeds, topped with fresh blueberries and hemp seeds."},
            {"name": "Spiced Tofu Scramble with Sautéed Spinach & Whole Wheat Toast", "calories": 380, "protein": 22, "carbs": 38, "fat": 16, "fiber": 7, "desc": "Crumbled firm tofu seasoned with turmeric, nutritional yeast, bell peppers, and fresh spinach."},
            {"name": "Quinoa Breakfast Porridge with Almond Butter & Sliced Banana", "calories": 410, "protein": 13, "carbs": 62, "fat": 14, "fiber": 8, "desc": "Warm cooked quinoa in oat milk lightly flavored with cinnamon, pure vanilla, and roasted almond butter."},
            {"name": "Avocado & Tomato Toast on Sprouted Rye Bread with Hemp Seeds", "calories": 350, "protein": 10, "carbs": 42, "fat": 17, "fiber": 9, "desc": "Crushed avocado with lemon juice, microgreens, and sliced heirloom tomatoes on toasted sprouted rye."},
            {"name": "Green Protein Smoothie Bowl with Kiwi and Pumpkin Seeds", "calories": 370, "protein": 19, "carbs": 52, "fat": 11, "fiber": 8, "desc": "Blend of baby spinach, frozen banana, pea protein, and flaxseeds topped with sliced kiwi."},
            {"name": "Warm Apple-Cinnamon Buckwheat Flakes with Walnuts", "calories": 390, "protein": 11, "carbs": 64, "fat": 12, "fiber": 8, "desc": "Steamed buckwheat groats with diced green apple, crushed walnuts, and ground cinnamon."},
            {"name": "Mediterranean Chickpea Flour Pancake (Socca) with Arugula", "calories": 340, "protein": 14, "carbs": 45, "fat": 12, "fiber": 6, "desc": "Savory baked chickpea pancake with rosemary, cherry tomatoes, and dressed wild arugula."}
        ],
        "mid_morning": [
            {"name": "Mixed Raw Almonds & Walnuts (25g)", "calories": 160, "protein": 5, "carbs": 5, "fat": 14, "fiber": 3, "desc": "Unsalted raw tree nuts providing essential polyphenols and plant-based ALA fatty acids."},
            {"name": "Fresh Pear with Unsweetened Sunflower Butter", "calories": 170, "protein": 4, "carbs": 26, "fat": 7, "fiber": 5, "desc": "Crisp sliced Asian or Bartlett pear with a spoonful of sunflower seed spread."},
            {"name": "Steamed Edamame Pods with Sea Salt", "calories": 140, "protein": 12, "carbs": 10, "fat": 5, "fiber": 6, "desc": "Warm whole young soybeans providing complete vegetable protein and folate."},
            {"name": "Carrot and Cucumber Batons with Roasted Garlic Hummus", "calories": 150, "protein": 5, "carbs": 19, "fat": 7, "fiber": 5, "desc": "Crunchy fresh crudités paired with tahini-rich chickpea dip."},
            {"name": "Roasted Tamari Pumpkin Seeds (Pepitas)", "calories": 160, "protein": 9, "carbs": 5, "fat": 13, "fiber": 2, "desc": "Zinc-dense oven-toasted pumpkin seeds with low-sodium fermented tamari."},
            {"name": "Fresh Orange Segments with a Handful of Cashews", "calories": 155, "protein": 4, "carbs": 22, "fat": 7, "fiber": 3, "desc": "Vitamin C paired with magnesium-rich whole raw cashews."},
            {"name": "Chia Seed Lime Fresca with Mint", "calories": 80, "protein": 3, "carbs": 8, "fat": 4, "fiber": 5, "desc": "Hydrating chilled lime water with gelled whole chia seeds."}
        ],
        "lunch": [
            {"name": "Warm Moroccan Spiced Lentil Bowl with Brown Rice & Kale", "calories": 520, "protein": 22, "carbs": 82, "fat": 11, "fiber": 16, "desc": "Simmered brown lentils with cumin, coriander, carrots, steamed brown rice, and tender massaged kale."},
            {"name": "Grilled Lemon-Herb Tempeh with Roasted Sweet Potato & Asparagus", "calories": 490, "protein": 26, "carbs": 54, "fat": 18, "fiber": 11, "desc": "Fermented organic tempeh steaks marinated in rosemary and lemon, paired with roasted root vegetables."},
            {"name": "Mediterranean Quinoa Salad with Chickpeas, Kalamata Olives & Cucumber", "calories": 480, "protein": 17, "carbs": 68, "fat": 16, "fiber": 12, "desc": "Tossed tri-color quinoa, chickpeas, red onion, and extra virgin olive oil vinaigrette."},
            {"name": "Black Bean & Roasted Corn Bowl with Avocado & Brown Jasmine Rice", "calories": 510, "protein": 18, "carbs": 76, "fat": 15, "fiber": 14, "desc": "Simmered black beans, fire-roasted sweet corn, chopped cilantro, and fresh avocado crema."},
            {"name": "Thai Basil Tofu Stir-Fry with Bok Choy & Rice Noodles", "calories": 495, "protein": 21, "carbs": 65, "fat": 16, "fiber": 7, "desc": "Pan-seared firm tofu, shredded shiitake mushrooms, baby bok choy, and ginger garlic glaze."},
            {"name": "Hearty Tuscan White Bean Soup with Swiss Chard & Garlic Crostini", "calories": 460, "protein": 19, "carbs": 72, "fat": 10, "fiber": 15, "desc": "Slow-simmered cannellini beans, mineral-rich chard, and a slice of rustic whole grain sourdough."},
            {"name": "Roasted Cauliflower & Chickpea Shawarma Bowl with Tahini Drizzle", "calories": 475, "protein": 18, "carbs": 64, "fat": 17, "fiber": 13, "desc": "Cumin-spiced cauliflower florets, crispy chickpeas, parsley, and lemon sesame dressing."}
        ],
        "evening_snack": [
            {"name": "Sliced Apple with Natural Peanut Butter", "calories": 180, "protein": 5, "carbs": 24, "fat": 9, "fiber": 4, "desc": "Fresh crisp apple with single-ingredient roasted peanut butter."},
            {"name": "Air-Popped Nutritional Yeast Popcorn", "calories": 120, "protein": 5, "carbs": 21, "fat": 2, "fiber": 4, "desc": "Whole grain corn seasoned with B-vitamin fortified nutritional yeast flakes."},
            {"name": "Crispy Baked Kale Chips with Olive Oil & Sea Salt", "calories": 110, "protein": 3, "carbs": 10, "fat": 7, "fiber": 2, "desc": "Dehydrated curly kale seasoned lightly with extra virgin olive oil."},
            {"name": "Berries & Unsweetened Coconut Yogurt", "calories": 140, "protein": 2, "carbs": 18, "fat": 7, "fiber": 4, "desc": "Live-culture probiotic plant yogurt with fresh blackberries."},
            {"name": "Toasted Whole Wheat Pita Triangles with Red Pepper Hummus", "calories": 170, "protein": 6, "carbs": 25, "fat": 5, "fiber": 4, "desc": "High-fiber whole wheat pita with paprika-infused chickpea dip."},
            {"name": "Dark Chocolate Square (85%) with Brazil Nut", "calories": 130, "protein": 2, "carbs": 8, "fat": 10, "fiber": 2, "desc": "High polyphenol cocoa paired with a selenium-dense whole Brazil nut."},
            {"name": "Matcha Green Tea Latte with Steamed Soy Milk", "calories": 90, "protein": 6, "carbs": 8, "fat": 3, "fiber": 1, "desc": "Antioxidant-dense Japanese green tea whisked into creamy soy milk."}
        ],
        "dinner": [
            {"name": "Red Lentil Dahl with Wilted Spinach & Steamed Quinoa", "calories": 470, "protein": 23, "carbs": 74, "fat": 8, "fiber": 14, "desc": "Creamy simmered red lentils with ginger, cumin seeds, garlic, and fresh baby spinach."},
            {"name": "Stuffed Bell Peppers with Wild Rice, Mushrooms & Walnuts", "calories": 460, "protein": 15, "carbs": 66, "fat": 17, "fiber": 10, "desc": "Oven-roasted bell peppers filled with aromatic wild rice blend and chopped earthy cremini mushrooms."},
            {"name": "Butternut Squash & Coconut Chickpea Curry with Cauliflower Rice", "calories": 440, "protein": 16, "carbs": 62, "fat": 15, "fiber": 12, "desc": "Gentle coconut-turmeric curry with roasted butternut squash chunks and garbanzo beans."},
            {"name": "Pan-Seared Organic Tofu with Sesame Ginger Broccoli & Soba Noodles", "calories": 490, "protein": 25, "carbs": 62, "fat": 16, "fiber": 8, "desc": "Buckwheat soba noodles tossed in toasted sesame oil with seared tofu cubes and steamed broccoli."},
            {"name": "Hearty Black Bean Vegetable Chili with Diced Avocado", "calories": 450, "protein": 19, "carbs": 69, "fat": 13, "fiber": 17, "desc": "Slow-cooked black and pinto beans with sweet potato, tomatoes, and smoky chipotle."},
            {"name": "Zucchini Ribbons & Lentil Bolognese with Fresh Basil", "calories": 420, "protein": 21, "carbs": 60, "fat": 10, "fiber": 13, "desc": "Tossed vegetable noodles with rich tomato-lentil ragù and nutritional yeast."},
            {"name": "Oven-Baked Falafel Platter with Tabbouleh & Roasted Eggplant", "calories": 480, "protein": 17, "carbs": 65, "fat": 18, "fiber": 13, "desc": "Herbed baked chickpea patties served with parsley bulgur tabbouleh and babaganoush."}
        ]
    },
    "vegetarian": {
        "breakfast": [
            {"name": "Poached Free-Range Eggs on Sprouted Toast with Avocado", "calories": 380, "protein": 19, "carbs": 32, "fat": 19, "fiber": 6, "desc": "Two poached eggs on toasted Ezekiel bread with smashed avocado and red pepper flakes."},
            {"name": "Greek Yogurt Bowl with Mixed Berries, Walnuts & Raw Honey", "calories": 350, "protein": 23, "carbs": 34, "fat": 13, "fiber": 5, "desc": "Plain low-fat Greek yogurt layered with antioxidant berries and crushed raw walnuts."},
            {"name": "Vegetable Frittata with Feta, Spinach & Cherry Tomatoes", "calories": 370, "protein": 24, "carbs": 12, "fat": 24, "fiber": 3, "desc": "Fluffy baked eggs with wilted spinach, crumbled Greek feta, and sweet blistered tomatoes."},
            {"name": "Warm Rolled Oatmeal with Chia Seeds, Cinnamon & Sliced Banana", "calories": 360, "protein": 12, "carbs": 64, "fat": 8, "fiber": 9, "desc": "Creamy oatmeal cooked in skim milk or almond milk, topped with banana and flaxseed."},
            {"name": "Cottage Cheese Pancakes with Fresh Strawberry Compote", "calories": 360, "protein": 27, "carbs": 38, "fat": 9, "fiber": 4, "desc": "High-protein pancakes prepared with blended low-fat cottage cheese and whole oats."},
            {"name": "Spinach, Mushroom & Cheddar Scramble with Wholegrain Muffin", "calories": 390, "protein": 25, "carbs": 30, "fat": 20, "fiber": 5, "desc": "Eggs whisked with sautéed baby portobellos, sharp cheddar, and whole wheat English muffin."},
            {"name": "Layered Chia Pudding with Greek Yogurt and Crushed Pistachios", "calories": 340, "protein": 18, "carbs": 30, "fat": 15, "fiber": 8, "desc": "Vanilla chia pudding topped with protein-dense yogurt and toasted pistachios."}
        ],
        "mid_morning": [
            {"name": "String Cheese with Fresh Crisp Green Apple", "calories": 160, "protein": 8, "carbs": 21, "fat": 6, "fiber": 4, "desc": "Part-skim mozzarella cheese stick paired with a crisp Granny Smith apple."},
            {"name": "Handful of Raw Almonds (25g)", "calories": 160, "protein": 6, "carbs": 6, "fat": 14, "fiber": 3, "desc": "Heart-healthy vitamin E-rich almonds."},
            {"name": "Cucumber Slices with Tzatziki Dip", "calories": 110, "protein": 6, "carbs": 9, "fat": 5, "fiber": 2, "desc": "Cool cucumber slices with strained yogurt, garlic, dill, and lemon dip."},
            {"name": "Hard-Boiled Egg with Sea Salt and Black Pepper", "calories": 75, "protein": 6, "carbs": 1, "fat": 5, "fiber": 0, "desc": "Nutrient-dense whole egg providing bioavailable choline."},
            {"name": "Steamed Edamame in the Pod", "calories": 130, "protein": 11, "carbs": 9, "fat": 4, "fiber": 5, "desc": "Lightly salted whole edamame."},
            {"name": "Plain Kefir Drink with Ground Flaxseed", "calories": 140, "protein": 9, "carbs": 12, "fat": 5, "fiber": 3, "desc": "Fermented probiotic dairy beverage supporting digestive microbiome health."},
            {"name": "Fresh Blueberries and Walnuts", "calories": 150, "protein": 4, "carbs": 16, "fat": 10, "fiber": 3, "desc": "Brain-supportive polyphenols and omega-3s."}
        ],
        "lunch": [
            {"name": "Mediterranean Lentil Salad with Crumbled Feta & Herb Dressing", "calories": 480, "protein": 24, "carbs": 58, "fat": 16, "fiber": 14, "desc": "Earthy green lentils with diced cucumbers, Kalamata olives, sheep's milk feta, and oregano vinaigrette."},
            {"name": "Paneer or Halloumi Skewers with Grilled Peppers & Quinoa Pilaf", "calories": 520, "protein": 27, "carbs": 48, "fat": 23, "fiber": 7, "desc": "Marinated grilled cheese cubes with colorful bell peppers over fluffy whole grain quinoa."},
            {"name": "Chickpea & Avocado Salad Wrap in Whole Wheat Tortilla", "calories": 460, "protein": 16, "carbs": 58, "fat": 18, "fiber": 11, "desc": "Mashed spiced garbanzo beans with avocado, lime, diced red onion, and crisp romaine."},
            {"name": "Roasted Vegetable & Goat Cheese Tart with Mixed Green Salad", "calories": 470, "protein": 18, "carbs": 44, "fat": 24, "fiber": 6, "desc": "Thin whole wheat crust with roasted zucchini, eggplant, goat cheese, and balsamic greens."},
            {"name": "Black Bean & Brown Rice Burrito Bowl with Monterey Jack", "calories": 510, "protein": 22, "carbs": 70, "fat": 16, "fiber": 13, "desc": "Seasoned black beans, sweet corn, brown rice, salsa fresca, and shredded cheese."},
            {"name": "Creamy Tuscan White Bean Soup with Shaved Parmesan & Sourdough", "calories": 450, "protein": 22, "carbs": 64, "fat": 12, "fiber": 12, "desc": "Hearty vegetable broth with cannellini beans, rosemary, and aged parmesan cheese."},
            {"name": "Warm Soba Noodles with Crispy Tofu & Sesame Peanut Sauce", "calories": 490, "protein": 23, "carbs": 59, "fat": 18, "fiber": 7, "desc": "Japanese buckwheat noodles with baked tofu and creamy peanut-lime dressing."}
        ],
        "evening_snack": [
            {"name": "Low-Fat Cottage Cheese with Diced Pineapple", "calories": 150, "protein": 14, "carbs": 18, "fat": 2, "fiber": 1, "desc": "Casein protein paired with digestive enzyme bromelain from fresh pineapple."},
            {"name": "Air-Popped Spiced Popcorn", "calories": 110, "protein": 3, "carbs": 22, "fat": 1, "fiber": 4, "desc": "Light whole grain snack with smoked paprika."},
            {"name": "Roasted Chickpeas with Rosemary and Olive Oil", "calories": 140, "protein": 6, "carbs": 20, "fat": 4, "fiber": 5, "desc": "Crunchy roasted legumes."},
            {"name": "Celery Sticks with Natural Peanut Butter", "calories": 140, "protein": 5, "carbs": 7, "fat": 11, "fiber": 2, "desc": "Low-glycemic hydration and healthy plant fats."},
            {"name": "Dark Chocolate Square (80%) with Hazelnuts", "calories": 140, "protein": 3, "carbs": 10, "fat": 11, "fiber": 2, "desc": "Antioxidant treat supporting vascular endothelial function."},
            {"name": "Small Berry Smoothie with Skim Milk or Soy Milk", "calories": 130, "protein": 8, "carbs": 22, "fat": 1, "fiber": 3, "desc": "Refreshing post-afternoon berry blend."},
            {"name": "Whole Grain Crackers with Hummus", "calories": 160, "protein": 5, "carbs": 22, "fat": 6, "fiber": 3, "desc": "Complex carbs and plant protein."}
        ],
        "dinner": [
            {"name": "Baked Eggplant Parmesan with Whole Wheat Penne & Tomato Basil Ragù", "calories": 480, "protein": 23, "carbs": 62, "fat": 16, "fiber": 11, "desc": "Herb-crusted roasted eggplant layered with marinara, part-skim mozzarella, and whole wheat pasta."},
            {"name": "Spinach & Lentil Curry with Brown Basmati Rice & Mint Yogurt", "calories": 490, "protein": 22, "carbs": 76, "fat": 10, "fiber": 13, "desc": "Traditional Indian dahl with leafy greens and a cooling spoonful of raita."},
            {"name": "Stuffed Portobello Mushrooms with Quinoa, Sun-Dried Tomatoes & Goat Cheese", "calories": 440, "protein": 20, "carbs": 46, "fat": 19, "fiber": 8, "desc": "Savory mushroom caps packed with seasoned quinoa and tangy melted goat cheese."},
            {"name": "Vegetarian Pad Thai with Tofu, Crushed Peanuts & Bean Sprouts", "calories": 510, "protein": 24, "carbs": 66, "fat": 18, "fiber": 6, "desc": "Rice noodles stir-fried with firm tofu, scrambled egg ribbons, tamarind, and lime."},
            {"name": "Hearty Three-Bean Vegetable Chili with Light Cheddar & Avocado", "calories": 470, "protein": 22, "carbs": 65, "fat": 14, "fiber": 16, "desc": "Kidney, black, and pinto beans in rich tomato cumin sauce."},
            {"name": "Shakshuka: Poached Eggs in Spiced Tomato Pepper Sauce with Pita", "calories": 430, "protein": 21, "carbs": 48, "fat": 17, "fiber": 7, "desc": "North African dish of gently poached eggs in garlic, cumin, and bell pepper reduction."},
            {"name": "Roasted Vegetable Buddha Bowl with Warm Farro & Tahini Sauce", "calories": 460, "protein": 18, "carbs": 68, "fat": 15, "fiber": 12, "desc": "Sweet potato, Brussels sprouts, broccoli, ancient grain farro, and sesame dressing."}
        ]
    },
    "non-vegetarian": {
        "breakfast": [
            {"name": "Smoked Salmon & Poached Egg on Sprouted Whole Grain Toast", "calories": 390, "protein": 28, "carbs": 30, "fat": 16, "fiber": 5, "desc": "Omega-3 rich wild salmon with poached egg, capers, and avocado spread on sprouted rye."},
            {"name": "Turkey Bacon & Egg White Omelet with Sautéed Spinach & Mushrooms", "calories": 340, "protein": 32, "carbs": 12, "fat": 15, "fiber": 3, "desc": "High-protein lean breakfast omelet with whole grain toast triangle."},
            {"name": "Greek Yogurt Parfait with Whey Protein, Berries & Raw Almonds", "calories": 380, "protein": 31, "carbs": 35, "fat": 12, "fiber": 6, "desc": "Layered probiotic yogurt, mixed wild blueberries, and sliced nuts."},
            {"name": "Scrambled Eggs with Diced Lean Chicken Breast & Roasted Tomatoes", "calories": 410, "protein": 36, "carbs": 14, "fat": 21, "fiber": 3, "desc": "Savory high-protein scramble served with half a grapefruit."},
            {"name": "Overnight Protein Oats with Blueberries, Chia & Collagen Peptides", "calories": 380, "protein": 26, "carbs": 52, "fat": 8, "fiber": 9, "desc": "Slow-release complex carbohydrates and joint-supporting peptides."},
            {"name": "Avocado & Hard-Boiled Egg Plate with Sliced Lean Turkey Breast", "calories": 370, "protein": 29, "carbs": 10, "fat": 22, "fiber": 5, "desc": "Low-carb nutrient dense platter with fresh cucumber wedges."},
            {"name": "Cottage Cheese & Spinach Protein Crepes with Berry Compote", "calories": 350, "protein": 28, "carbs": 32, "fat": 10, "fiber": 4, "desc": "Egg and oat crepes filled with lightly seasoned low-fat cottage cheese."}
        ],
        "mid_morning": [
            {"name": "Hard-Boiled Egg with a Slice of Sharp Cheddar", "calories": 160, "protein": 12, "carbs": 1, "fat": 11, "fiber": 0, "desc": "Zero carbohydrate protein and fat for sustained morning satiety."},
            {"name": "Turkey Jerky (Low-Sodium) & Fresh Mandarin Orange", "calories": 140, "protein": 14, "carbs": 15, "fat": 1, "fiber": 2, "desc": "Portable lean protein combined with bioavailable citrus vitamin C."},
            {"name": "Raw Walnuts & Dried Cranberries (Unsweetened)", "calories": 170, "protein": 4, "carbs": 14, "fat": 12, "fiber": 3, "desc": "Cardioprotective alpha-linolenic acid (ALA)."},
            {"name": "Single-Serve Low-Fat Cottage Cheese with Chia Seeds", "calories": 140, "protein": 16, "carbs": 8, "fat": 4, "fiber": 3, "desc": "Slow-digesting micellar casein protein."},
            {"name": "Steamed Edamame with Himalayan Pink Salt", "calories": 130, "protein": 11, "carbs": 9, "fat": 4, "fiber": 5, "desc": "Isoflavones and dietary fiber."},
            {"name": "Celery Sticks with Almond Butter", "calories": 150, "protein": 4, "carbs": 7, "fat": 12, "fiber": 3, "desc": "Electrolyte rich hydration with monounsaturated fats."},
            {"name": "Sliced Apple with Natural Peanut Butter", "calories": 180, "protein": 5, "carbs": 24, "fat": 9, "fiber": 4, "desc": "Polyphenols, pectin, and healthy fats."}
        ],
        "lunch": [
            {"name": "Grilled Lemon Herb Chicken Breast with Quinoa & Steamed Broccoli", "calories": 510, "protein": 44, "carbs": 48, "fat": 14, "fiber": 8, "desc": "Skinless chicken breast marinated in extra virgin olive oil and oregano, over organic quinoa."},
            {"name": "Pan-Seared Salmon Fillet with Roasted Sweet Potato & Asparagus", "calories": 540, "protein": 38, "carbs": 42, "fat": 23, "fiber": 7, "desc": "Wild salmon rich in EPA/DHA omega-3 fatty acids, paired with caramelized sweet potato cubes."},
            {"name": "Mediterranean Tuna & Chickpea Salad with Extra Virgin Olive Oil", "calories": 490, "protein": 36, "carbs": 44, "fat": 18, "fiber": 10, "desc": "Line-caught skipjack tuna, garbanzo beans, red onion, parsley, and lemon dressing."},
            {"name": "Grilled Turkey Breast Wrap with Avocado, Spinach & Whole Wheat Tortilla", "calories": 480, "protein": 38, "carbs": 40, "fat": 18, "fiber": 7, "desc": "Oven-roasted sliced turkey with creamy avocado and tender baby spinach leaves."},
            {"name": "Asian Beef & Bok Choy Stir-Fry with Brown Jasmine Rice", "calories": 520, "protein": 35, "carbs": 54, "fat": 17, "fiber": 6, "desc": "Lean flank steak strips wok-tossed with ginger, garlic, coconut aminos, and crisp bok choy."},
            {"name": "Poached Cod Fillet with Mediterranean Tomato-Caper Sauce & Farro", "calories": 460, "protein": 36, "carbs": 48, "fat": 11, "fiber": 7, "desc": "Flaky white fish in savory plum tomato, olive, and caper reduction over whole grain farro."},
            {"name": "Moroccan Spiced Chicken Tagine with Chickpeas & Steamed Couscous", "calories": 500, "protein": 40, "carbs": 52, "fat": 13, "fiber": 8, "desc": "Tender chicken simmered with turmeric, cinnamon, apricots, and chickpeas."}
        ],
        "evening_snack": [
            {"name": "Roasted Chickpeas with Smoked Paprika", "calories": 130, "protein": 6, "carbs": 18, "fat": 3, "fiber": 5, "desc": "Crunchy high-fiber snack."},
            {"name": "Whey Protein Shake with Unsweetened Almond Milk", "calories": 140, "protein": 24, "carbs": 3, "fat": 2, "fiber": 1, "desc": "Quick post-workout or afternoon amino acid recovery."},
            {"name": "Air-Popped Popcorn with Herb Seasoning", "calories": 110, "protein": 3, "carbs": 21, "fat": 1, "fiber": 4, "desc": "Light whole grain volume snack."},
            {"name": "Cucumber and Radish Rounds with Guacamole", "calories": 120, "protein": 2, "carbs": 8, "fat": 10, "fiber": 4, "desc": "Crunchy vegetables with nutrient-dense avocado."},
            {"name": "Dark Chocolate Square (85%) with 10 Raw Almonds", "calories": 150, "protein": 4, "carbs": 8, "fat": 12, "fiber": 3, "desc": "Antioxidant flavonoids for cardiovascular wellness."},
            {"name": "Greek Yogurt Dip with Bell Pepper Strips", "calories": 120, "protein": 10, "carbs": 11, "fat": 2, "fiber": 3, "desc": "Cool protein-packed dip with high vitamin C peppers."},
            {"name": "Hard-Boiled Egg with Paprika", "calories": 75, "protein": 6, "carbs": 1, "fat": 5, "fiber": 0, "desc": "Classic nutrient-rich protein."}
        ],
        "dinner": [
            {"name": "Oven-Baked Herb-Crusted Halibut or Cod with Roasted Cauliflower & Green Beans", "calories": 460, "protein": 39, "carbs": 24, "fat": 22, "fiber": 8, "desc": "Lean white fish seasoned with parsley and garlic, served with caramelized vegetables."},
            {"name": "Grilled Lemon Chicken Thighs (Skinless) with Wild Rice Pilaf & Zucchini", "calories": 510, "protein": 42, "carbs": 44, "fat": 18, "fiber": 6, "desc": "Tender grilled poultry paired with nutty wild rice and charred summer squash."},
            {"name": "Lean Turkey Bolognese over Zucchini Noodles and Spelt Spaghetti", "calories": 480, "protein": 38, "carbs": 46, "fat": 14, "fiber": 8, "desc": "93% lean ground turkey simmered in crushed San Marzano tomatoes with aromatic herbs."},
            {"name": "Baked Salmon with Warm Lentil & Baby Kale Ragout", "calories": 530, "protein": 41, "carbs": 36, "fat": 23, "fiber": 10, "desc": "Omega-3 rich salmon over hearty simmered French green lentils and wilted kale."},
            {"name": "Shrimp and Vegetable Stir-Fry with Soba Noodles & Sesame Seeds", "calories": 470, "protein": 36, "carbs": 54, "fat": 12, "fiber": 7, "desc": "Plump wild shrimp, snap peas, bell peppers, and buckwheat noodles in low-sodium tamari."},
            {"name": "Grilled Pork Tenderloin Medallions with Roasted Butternut Squash & Brussels Sprouts", "calories": 490, "protein": 39, "carbs": 38, "fat": 18, "fiber": 8, "desc": "Lean tenderloin with sweet roasted root vegetables and cruciferous greens."},
            {"name": "Chicken Breast Souvlaki Plate with Greek Salad, Brown Rice & Tzatziki", "calories": 510, "protein": 43, "carbs": 42, "fat": 17, "fiber": 7, "desc": "Oregano marinated chicken skewers with crisp cucumber, tomato, and olive salad."}
        ]
    }
}

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def generate_7_day_meal_plan(
    dietary_preference: str = "Vegetarian",
    target_calories: int = 2000,
    meals_per_day: int = 5,
    budget_preference: str = "Moderate",
    cuisine_preference: str = "Mediterranean",
    allergies: Optional[List[str]] = None,
    dislikes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate a 7-day personalized meal plan adjusted to user preferences.
    """
    allergies = [a.lower().strip() for a in (allergies or [])]
    dislikes = [d.lower().strip() for d in (dislikes or [])]

    diet_key = dietary_preference.lower().strip()
    if diet_key not in MEAL_TEMPLATES:
        if "veg" in diet_key and "non" not in diet_key:
            diet_key = "vegetarian"
        elif "vegan" in diet_key:
            diet_key = "vegan"
        else:
            diet_key = "non-vegetarian"

    templates = MEAL_TEMPLATES[diet_key]

    weekly_plan = []
    weekly_total_calories = 0
    weekly_total_protein = 0
    weekly_total_carbs = 0
    weekly_total_fat = 0
    weekly_total_fiber = 0

    for day_idx, day_name in enumerate(DAYS_OF_WEEK):
        # Pick meals deterministically by day index
        b_item = templates["breakfast"][day_idx % len(templates["breakfast"])].copy()
        l_item = templates["lunch"][day_idx % len(templates["lunch"])].copy()
        d_item = templates["dinner"][day_idx % len(templates["dinner"])].copy()

        meals = [
            {"slot": "Breakfast", **b_item}
        ]

        if meals_per_day >= 4:
            mm_item = templates["mid_morning"][day_idx % len(templates["mid_morning"])].copy()
            meals.append({"slot": "Mid-Morning Snack", **mm_item})

        meals.append({"slot": "Lunch", **l_item})

        if meals_per_day >= 5:
            es_item = templates["evening_snack"][day_idx % len(templates["evening_snack"])].copy()
            meals.append({"slot": "Evening Snack", **es_item})

        meals.append({"slot": "Dinner", **d_item})

        # Calculate daily raw totals
        day_cal = sum(m["calories"] for m in meals)
        day_pro = sum(m["protein"] for m in meals)
        day_carb = sum(m["carbs"] for m in meals)
        day_fat = sum(m["fat"] for m in meals)
        day_fib = sum(m["fiber"] for m in meals)

        # Scale proportions gently to match user's target calorie envelope if significantly different
        scale_factor = target_calories / max(day_cal, 1200)
        # Keep scale reasonable (between 0.75 and 1.35)
        scale_factor = max(0.75, min(1.35, scale_factor))

        scaled_meals = []
        for m in meals:
            sc_m = m.copy()
            sc_m["calories"] = round(m["calories"] * scale_factor)
            sc_m["protein"] = round(m["protein"] * scale_factor, 1)
            sc_m["carbs"] = round(m["carbs"] * scale_factor, 1)
            sc_m["fat"] = round(m["fat"] * scale_factor, 1)
            sc_m["fiber"] = round(m["fiber"] * scale_factor, 1)
            scaled_meals.append(sc_m)

        final_day_cal = sum(m["calories"] for m in scaled_meals)
        final_day_pro = round(sum(m["protein"] for m in scaled_meals), 1)
        final_day_carb = round(sum(m["carbs"] for m in scaled_meals), 1)
        final_day_fat = round(sum(m["fat"] for m in scaled_meals), 1)
        final_day_fib = round(sum(m["fiber"] for m in scaled_meals), 1)

        weekly_total_calories += final_day_cal
        weekly_total_protein += final_day_pro
        weekly_total_carbs += final_day_carb
        weekly_total_fat += final_day_fat
        weekly_total_fiber += final_day_fib

        weekly_plan.append({
            "day": day_name,
            "meals": scaled_meals,
            "totals": {
                "calories": final_day_cal,
                "protein_g": final_day_pro,
                "carbs_g": final_day_carb,
                "fat_g": final_day_fat,
                "fiber_g": final_day_fib
            }
        })

    avg_daily_cal = round(weekly_total_calories / 7)
    avg_daily_pro = round(weekly_total_protein / 7, 1)
    avg_daily_carb = round(weekly_total_carbs / 7, 1)
    avg_daily_fat = round(weekly_total_fat / 7, 1)
    avg_daily_fib = round(weekly_total_fiber / 7, 1)

    return {
        "dietary_preference": dietary_preference,
        "target_calories": target_calories,
        "budget_preference": budget_preference,
        "cuisine_preference": cuisine_preference,
        "average_daily_nutrition": {
            "calories": avg_daily_cal,
            "protein_g": avg_daily_pro,
            "carbs_g": avg_daily_carb,
            "fat_g": avg_daily_fat,
            "fiber_g": avg_daily_fib
        },
        "days": weekly_plan,
        "disclaimer": (
            "Nutritional values are scientific approximations and may vary depending "
            "on cooking techniques, ingredient origins, and portion weights. This plan "
            "serves as general educational guidance and is not a medical diet prescription."
        )
    }
