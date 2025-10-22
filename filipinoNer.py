import streamlit as st
import spacy
from spacy import displacy

# Load model just once!
@st.cache_resource
def load_model():
    return spacy.load("tl_calamancy_md-0.1.0")

nlp = load_model()

st.title("Combining NER and Streamlit!")
user_input = st.text_area("Enter text here:", height=200)

if st.button("Analyze!"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        doc = nlp(user_input)

        st.subheader("Named Entities:")
        if doc.ents:
            ent_count = 0
            for ent in doc.ents: 
                st.write(f"**{ent.text}** -> {ent.label_}")
                ent_count += 1
            st.write(f"\nTotal number of entities: {ent_count}\n")
        else:
            st.write("No named entities found.")
        
        st.subheader("\nTokenization and Attributes:")
        token_count = 0
        for token in doc:
            st.write(f"{token.text} -> {token.lemma_} -> {token.pos_} -> {token.tag_} -> {token.dep_} -> {token.shape_} -> {token.is_alpha} -> {token.is_stop}")
            token_count += 1
        st.write(f"\nTotal number of tokens: {token_count}\n")

        st.subheader("Highlighted Entities in Text:")
        html = displacy.render(doc, style="ent", page=True)
        st.components.v1.html(html, height=300, scrolling=True)
