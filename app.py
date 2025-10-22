# app.py
import streamlit as st
import spacy
from spacy import displacy
from filipino_food_config import (
    FilipinoFoodNER, 
    FILIPINO_FOODS, 
    FILIPINO_FOOD_VARIATIONS,
    DISPLAY_CONFIG, 
    APP_CONFIG
)

# Page configuration
st.set_page_config(
    page_title="Filipino Food NER",
    page_icon="🍽️",
    layout="wide"
)

# Load model just once using caching
@st.cache_resource
def load_filipino_food_model():
    """Load the Filipino Food NER model with caching."""
    ner_model = FilipinoFoodNER()
    return ner_model.load_model_with_ruler()

@st.cache_resource  
def get_sample_texts():
    """Get sample texts for testing."""
    ner_model = FilipinoFoodNER()
    return ner_model.get_sample_texts()

# Initialize the model
nlp = load_filipino_food_model()
sample_texts = get_sample_texts()

# Main app
def main():
    st.title(f"{APP_CONFIG['emoji']} {APP_CONFIG['title']} with Streamlit!")
    st.markdown(f"*{APP_CONFIG['description']}*")
    
    # Sidebar
    setup_sidebar()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        user_input = st.text_area(
            "Enter text here:", 
            height=200,
            help="Enter any text to analyze. The app will detect people, organizations, places, and Filipino foods!",
            placeholder="Try: 'John visited Manila and enjoyed Sinigang at the restaurant with Maria from Cebu.'"
        )
    
    with col2:
        st.subheader("Entity Types Detected")
        if user_input.strip():
            # Quick preview of all entity types
            doc_preview = nlp(user_input)
            entity_types = set(ent.label_ for ent in doc_preview.ents)
            
            if entity_types:
                st.success(f"Found {len(entity_types)} entity types")
                
                # Show Filipino food specifically
                filipino_foods = [ent.text for ent in doc_preview.ents if ent.label_ == "FILIPINO_FOOD"]
                if filipino_foods:
                    st.markdown(f"**{APP_CONFIG['emoji']} Filipino Foods:** {len(filipino_foods)}")
                
                # Show other entity types
                other_types = [t for t in entity_types if t != "FILIPINO_FOOD"]
                if other_types:
                    st.markdown(f"**Other Entities:** {', '.join(sorted(other_types))}")
                    
            else:
                st.info("No entities detected in preview")
        else:
            st.info("Enter text to see entity preview")
            st.markdown("""
            **This app detects:**
            - 🍽️ Filipino Foods
            - 👤 People (PERSON)
            - 🏢 Organizations (ORG) 
            - 🌍 Places (GPE)
            - 📅 Dates & Times
            - 💰 Money & Numbers
            - And more!
            """)
    
    # Analysis button
    if st.button("🔍 Analyze Text", type="primary", width="stretch"):
        analyze_text(user_input)

def analyze_text(user_input):
    """Analyze the input text for Filipino food entities."""
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
        return
    
    with st.spinner("Analyzing text..."):
        doc = nlp(user_input)
    
    # Results in tabs
    tab1, tab2, tab3 = st.tabs(["📊 All Entities", "🔍 Detailed Analysis", "🎨 Visualization"])
    
    with tab1:
        display_all_entities(doc)
    
    with tab2:
        display_detailed_analysis(doc)
    
    with tab3:
        display_visualization(doc)
    


def display_filipino_foods(doc):
    """Display Filipino food entities found in the text."""
    st.subheader("Filipino Foods Detected")
    
    filipino_foods_found = [ent for ent in doc.ents if ent.label_ == "FILIPINO_FOOD"]
    
    if filipino_foods_found:
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Filipino Foods", len(filipino_foods_found))
        with col2:
            unique_foods = len(set(ent.text for ent in filipino_foods_found))
            st.metric("Unique Dishes", unique_foods)
        with col3:
            coverage = (unique_foods / len(FILIPINO_FOODS)) * 100
            st.metric("Coverage %", f"{coverage:.1f}%")
        
        st.markdown("---")
        
        # List found foods
        st.subheader("Found Filipino Dishes:")
        unique_foods_list = list(set(ent.text for ent in filipino_foods_found))
        
        # Display in columns
        cols = st.columns(3)
        for i, food in enumerate(sorted(unique_foods_list)):
            with cols[i % 3]:
                st.write(f"{APP_CONFIG['emoji']} **{food}**")
        
        st.success(f"Great! Found {len(filipino_foods_found)} Filipino food mention(s) in your text! {APP_CONFIG['flag']}")
    else:
        st.info("No Filipino foods detected in the text. Try mentioning dishes like Sinigang, Adobo, or Lechon!")
        
        # Suggestions
        st.subheader("Popular Filipino Dishes to Try:")
        popular_dishes = ["Sinigang", "Adobo", "Lechon", "Kare-kare", "Pancit", "Halo-halo"]
        cols = st.columns(3)
        for i, dish in enumerate(popular_dishes):
            with cols[i % 3]:
                st.write(f"{APP_CONFIG['emoji']} {dish}")

def display_all_entities(doc):
    """Display all named entities found in the text."""
    st.subheader("All Named Entities Found")
    
    if doc.ents:
        # Group entities by label
        entity_groups = {}
        for ent in doc.ents:
            if ent.label_ not in entity_groups:
                entity_groups[ent.label_] = []
            entity_groups[ent.label_].append(ent.text)
        
        # Entity type descriptions
        entity_descriptions = {
            "FILIPINO_FOOD": "Filipino cuisine and dishes",
            "PERSON": "Names of people",
            "ORG": "Organizations, companies, agencies",
            "GPE": "Countries, cities, states",
            "DATE": "Dates and time expressions",
            "MONEY": "Monetary values",
            "CARDINAL": "Numbers",
            "ORDINAL": "Ordinal numbers (first, second, etc.)",
            "TIME": "Times of day",
            "PERCENT": "Percentages",
            "QUANTITY": "Measurements and quantities",
            "FAC": "Facilities and buildings",
            "PRODUCT": "Products and objects",
            "EVENT": "Named events",
            "WORK_OF_ART": "Titles of books, songs, etc.",
            "LAW": "Legal documents",
            "LANGUAGE": "Languages",
            "NORP": "Nationalities, religious groups"
        }
        
        # Sort entity groups - Filipino Food first, then alphabetical
        sorted_labels = sorted(entity_groups.keys())
        if "FILIPINO_FOOD" in sorted_labels:
            sorted_labels.remove("FILIPINO_FOOD")
            sorted_labels = ["FILIPINO_FOOD"] + sorted_labels
        
        # Display entities in columns
        col1, col2 = st.columns(2)
        
        for i, label in enumerate(sorted_labels):
            entities = entity_groups[label]
            unique_entities = list(set(entities))
            
            with col1 if i % 2 == 0 else col2:
                # Special styling for Filipino food
                if label == "FILIPINO_FOOD":
                    st.markdown(f"### {APP_CONFIG['emoji']} {label}")
                    st.markdown(f"*{entity_descriptions.get(label, 'Entity type')}*")
                else:
                    st.markdown(f"### {label}")
                    st.markdown(f"*{entity_descriptions.get(label, 'Entity type')}*")
                
                # Display entities
                for entity in unique_entities:
                    count = entities.count(entity)
                    if label == "FILIPINO_FOOD":
                        st.write(f"{APP_CONFIG['emoji']} **{entity}** {f'({count}x)' if count > 1 else ''}")
                    else:
                        st.write(f"• **{entity}** {f'({count}x)' if count > 1 else ''}")
                
                st.markdown("---")
        
        # Summary statistics
        st.subheader("Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Entities", len(doc.ents))
        with col2:
            st.metric("Entity Types", len(entity_groups))
        with col3:
            filipino_count = len(entity_groups.get("FILIPINO_FOOD", []))
            st.metric("Filipino Foods", filipino_count)
        with col4:
            other_count = len(doc.ents) - filipino_count
            st.metric("Other Entities", other_count)
            
    else:
        st.info("No named entities found in the text.")
        st.markdown("""
        **Try entering text that contains:**
        - People's names (John, Maria, Dr. Smith)
        - Places (Manila, Philippines, New York)
        - Organizations (Google, UN, Department of Health)
        - Filipino foods (Sinigang, Adobo, Lechon)
        - Dates and numbers
        """)

def display_detailed_analysis(doc):
    """Display detailed token analysis."""
    st.subheader("Detailed Token Analysis")
    
    # Token statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tokens", len(doc))
    with col2:
        alpha_tokens = sum(1 for token in doc if token.is_alpha)
        st.metric("Word Tokens", alpha_tokens)
    with col3:
        stop_words = sum(1 for token in doc if token.is_stop)
        st.metric("Stop Words", stop_words)
    with col4:
        entities = len(doc.ents)
        st.metric("Entities", entities)
    
    # Detailed token table
    with st.expander("Click to see detailed token analysis"):
        import pandas as pd
        
        token_data = []
        for token in doc:
            token_data.append({
                "Text": token.text,
                "Lemma": token.lemma_,
                "POS": token.pos_,
                "Tag": token.tag_,
                "Dependency": token.dep_,
                "Shape": token.shape_,
                "Is Alpha": token.is_alpha,
                "Is Stop": token.is_stop
            })
        
        df = pd.DataFrame(token_data)
        st.dataframe(df, width='stretch')

def display_visualization(doc):
    """Display the spaCy visualization."""
    st.subheader("Entity Visualization")
    
    try:
        html = displacy.render(
            doc, 
            style="ent", 
            options=DISPLAY_CONFIG["options"], 
            page=True
        )
        st.components.v1.html(html, height=400, scrolling=True)
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        st.write("Fallback: Entities found in text:")
        for ent in doc.ents:
            if ent.label_ == "FILIPINO_FOOD":
                st.write(f"{APP_CONFIG['emoji']} **{ent.text}** -> {ent.label_}")
            else:
                st.write(f"• **{ent.text}** -> {ent.label_}")

def setup_sidebar():
    """Setup the sidebar with information and controls."""
    st.sidebar.title("About This App")
    st.sidebar.markdown(f"""
    This app uses a custom NER model to identify Filipino food items in text.
    
    **Features:**
    - Recognizes {len(FILIPINO_FOODS)}+ Filipino dishes
    - Handles variations and alternative names
    - Real-time entity detection
    - Interactive visualizations
    """)
    
    # Model information
    st.sidebar.subheader("Model Info")
    st.sidebar.info(f"""
    **Base Model**: en_core_web_sm  
    **Custom Entities**: {len(FILIPINO_FOODS)} foods  
    **Variations**: {len(FILIPINO_FOOD_VARIATIONS)} alternative names
    """)
    
    # Food database
    with st.sidebar.expander("View Food Database"):
        st.write("**Main Categories:**")
        categories = {
            "Main Dishes": FILIPINO_FOODS[:15],
            "Desserts": FILIPINO_FOODS[35:43],
            "Street Food": FILIPINO_FOODS[43:51],
            "Others": FILIPINO_FOODS[51:]
        }
        
        for category, foods in categories.items():
            st.write(f"**{category}:**")
            for food in foods[:5]:  # Show first 5 of each category
                st.write(f"• {food}")
            if len(foods) > 5:
                st.write(f"... and {len(foods) - 5} more")
    
    # Instructions
    st.sidebar.subheader("How to Use")
    st.sidebar.markdown("""
    1. Choose a sample text or enter your own
    2. Click "Analyze Text" 
    3. View results in different tabs
    4. Check the visualization for highlighted entities
    """)
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        st.markdown("""
        **Want to extend this app?**
        - Add more foods to `FILIPINO_FOODS` list
        - Train a custom spaCy model for better accuracy
        - Add support for other cuisines
        - Implement confidence scoring
        """)

if __name__ == "__main__":
    main()