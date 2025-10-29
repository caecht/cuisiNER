# filipino_food_config.py
import spacy
from spacy.pipeline import EntityRuler
from spacy.training import Example
import random

# Comprehensive Filipino food items database
FILIPINO_FOODS = [
    # Main dishes
    "Sinigang", "Adobo", "Lechon", "Kare-kare", "Bicol Express", "Lumpia", 
    "Pancit", "Sisig", "Tinola", "Bulalo", "Menudo", "Caldereta",
    "Mechado", "Afritada", "Pakbet", "Pinakbet", "Laing", "Ginataang Bilo-bilo",
    "Tapa", "Tocino", "Longganisa", "Bangus", "Lechon Kawali", "Crispy Pata",
    "Kinilaw", "Inihaw", "Grilled Liempo", "Pork BBQ", "Chicken Inasal",
    "Palabok", "Mami", "Arroz Caldo", "Goto", "Lugaw", "Champorado",
    
    # Desserts
    "Halo-halo", "Biko", "Leche Flan", "Ube Halaya", "Maja Blanca", "Turon", 
    "Bibingka", "Puto", "Kutsinta", "Suman", "Taho", "Buko Pie",
    "Ensaymada", "Pan de Sal", "Monay", "Spanish Bread",
    
    # Street food
    "Balut", "Kwek-kwek", "Fish Balls", "Isaw", "Betamax", "Adidas", 
    "Helmet", "Chicharon",
    
    # Vegetables and fruits
    "Malunggay", "Ampalaya", "Okra", "Sitaw", "Kangkong", "Camote", 
    "Ube", "Lanzones", "Rambutan", "Mangosteen", "Durian", "Jackfruit", 
    "Buko", "Saging",
    
    # Beverages
    "Kapeng Barako", "Buko Juice", "Sago't Gulaman"
]

# Additional variations and alternative names
FILIPINO_FOOD_VARIATIONS = {
    "Pancit Canton": "Pancit",
    "Pancit Bihon": "Pancit", 
    "Pancit Malabon": "Pancit",
    "Adobong Manok": "Adobo",
    "Adobong Baboy": "Adobo",
    "Sinigang na Baboy": "Sinigang",
    "Sinigang na Hipon": "Sinigang",
    "Chicken Adobo": "Adobo",
    "Pork Adobo": "Adobo",
    "Lechon Belly": "Lechon",
    "Halo Halo": "Halo-halo",
    "Ginataang Gulay": "Ginataang Bilo-bilo",
    "Fish Ball": "Fish Balls",
}

class FilipinoFoodNER:
    def __init__(self, base_model="en_core_web_sm"):
        """Initialize the Filipino Food NER model."""
        self.base_model = base_model
        self.nlp = None
        
    def load_model_with_ruler(self):
        """Load spaCy model with EntityRuler for Filipino food recognition."""
        nlp = spacy.load(self.base_model)
        
        # Create entity ruler
        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = nlp.get_pipe("entity_ruler")
        
        # Create patterns for Filipino food
        patterns = []
        
        # Add main food items
        for food in FILIPINO_FOODS:
            patterns.append({"label": "FILIPINO_FOOD", "pattern": food})
            patterns.append({"label": "FILIPINO_FOOD", "pattern": food.lower()})
            patterns.append({"label": "FILIPINO_FOOD", "pattern": food.upper()})
        
        # Add variations
        for variation, main_food in FILIPINO_FOOD_VARIATIONS.items():
            patterns.append({"label": "FILIPINO_FOOD", "pattern": variation})
            patterns.append({"label": "FILIPINO_FOOD", "pattern": variation.lower()})
            patterns.append({"label": "FILIPINO_FOOD", "pattern": variation.upper()})
        
        ruler.add_patterns(patterns)
        self.nlp = nlp
        return nlp
    
    def create_training_data(self):
        """Create training data for custom NER model training."""
        training_data = [
            ("I love eating Sinigang with rice.", {"entities": [(14, 22, "FILIPINO_FOOD")]}),
            ("My favorite dish is Adobo.", {"entities": [(20, 25, "FILIPINO_FOOD")]}),
            ("We had Lechon at the party.", {"entities": [(7, 13, "FILIPINO_FOOD")]}),
            ("Kare-kare is a traditional Filipino stew.", {"entities": [(0, 9, "FILIPINO_FOOD")]}),
            ("I ordered Pancit and Lumpia for dinner.", {"entities": [(10, 16, "FILIPINO_FOOD"), (21, 27, "FILIPINO_FOOD")]}),
            ("Halo-halo is a popular Filipino dessert.", {"entities": [(0, 9, "FILIPINO_FOOD")]}),
            ("The restaurant serves delicious Sisig.", {"entities": [(32, 37, "FILIPINO_FOOD")]}),
            ("My grandmother makes the best Tinola.", {"entities": [(30, 36, "FILIPINO_FOOD")]}),
            ("For breakfast, I had Tapa and Tocino with garlic rice.", {"entities": [(21, 25, "FILIPINO_FOOD"), (30, 36, "FILIPINO_FOOD")]}),
            ("The Bicol Express was so spicy!", {"entities": [(4, 17, "FILIPINO_FOOD")]}),
            ("We enjoyed Balut and Fish Balls at the night market.", {"entities": [(11, 16, "FILIPINO_FOOD"), (21, 31, "FILIPINO_FOOD")]}),
            ("My mom's Menudo recipe is the best.", {"entities": [(9, 15, "FILIPINO_FOOD")]}),
            ("I tried Durian for the first time in Davao.", {"entities": [(8, 14, "FILIPINO_FOOD")]}),
            ("The Lechon Kawali was crispy and delicious.", {"entities": [(4, 17, "FILIPINO_FOOD")]}),
            ("We had Arroz Caldo when it was raining.", {"entities": [(7, 18, "FILIPINO_FOOD")]}),
            ("Turon with ice cream is my favorite dessert.", {"entities": [(0, 5, "FILIPINO_FOOD")]}),
            ("The street vendor sells Kwek-kwek and Isaw.", {"entities": [(24, 33, "FILIPINO_FOOD"), (38, 42, "FILIPINO_FOOD")]}),
            ("Buko Pie from Laguna is famous.", {"entities": [(0, 8, "FILIPINO_FOOD")]}),
            ("I drink Kapeng Barako every morning.", {"entities": [(8, 21, "FILIPINO_FOOD")]}),
            ("The Pinakbet has fresh vegetables.", {"entities": [(4, 12, "FILIPINO_FOOD")]}),
        ]
        return training_data
    
    def train_custom_model(self, training_data, iterations=30, output_dir="./filipino_food_model"):
        """Train a custom NER model (advanced option)."""
        if self.nlp is None:
            self.nlp = spacy.load(self.base_model)
        
        # Add NER component if not present
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner")
        else:
            ner = self.nlp.get_pipe("ner")
        
        # Add new label
        ner.add_label("FILIPINO_FOOD")
        
        # Convert training data to spaCy format
        examples = []
        for text, annotations in training_data:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        
        # Training loop
        self.nlp.initialize()
        for i in range(iterations):
            random.shuffle(examples)
            losses = {}
            batches = spacy.util.minibatch(examples, size=8)
            for batch in batches:
                self.nlp.update(batch, losses=losses, drop=0.3)
            print(f"Iteration {i+1}, Loss: {losses}")
        
        # Save model
        self.nlp.to_disk(output_dir)
        print(f"Model saved to {output_dir}")
        
        return self.nlp
    
    def get_sample_texts(self):
        """Return sample texts for testing."""
        return [
            "Yesterday, I had delicious Sinigang na Baboy for lunch and Adobong Manok for dinner. We also enjoyed some Lechon and Halo-halo for dessert.",
            "My friend from America tried Balut for the first time and loved the Pancit Canton we prepared.",
            "The Filipino restaurant serves authentic Kare-kare, Bicol Express, and fresh Lumpia.",
            "For merienda, we had Turon, Biko, and Taho from the street vendor.",
            "The party menu included Lechon Kawali, Sisig, Crispy Pata, and various Filipino desserts like Leche Flan and Ube Halaya.",
        ]

# Display configuration
DISPLAY_CONFIG = {
    "colors": {"FILIPINO_FOOD": "#ff6b6b"},  # Red color for Filipino food
    "options": {
        "ents": ["PERSON", "ORG", "GPE", "FILIPINO_FOOD"], 
        "colors": {"FILIPINO_FOOD": "#ff6b6b"},
        "ents_by_label": {
            "FILIPINO_FOOD": {"color": "#ff6b6b", "textColor": "#ffffff"},
            "PERSON": {"textColor": "#ffffff"},
            "ORG": {"textColor": "#ffffff"},
            "GPE": {"textColor": "#ffffff"}
        }
    }
}

# App configuration
APP_CONFIG = {
    "title": "CuisiNER Filipino Food NER Recognition",
    "description": "This app can recognize Filipino food items like Sinigang, Adobo, Lechon, and many more!",
    "emoji": "🍽️",
    "flag": "🇵🇭"
}