import random
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
from tqdm import tqdm
import os
import json

MODEL_DIR = "custom_ner_model"

def create_docbin(training_data):
    """
    Convert training data to a DocBin for faster training.
    training_data: list of dicts with 'text' and 'entities' as [start, end, label]
    """
    nlp = spacy.blank("en")
    doc_bin = DocBin()

    for example in tqdm(training_data, desc="Creating DocBin"):
        text = example["text"]
        labels = example["entities"]
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in labels:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print(f"Skipping misaligned entity: '{text[start:end]}'")
            else:
                ents.append(span)
        filtered_ents = filter_spans(ents)
        doc.ents = filtered_ents
        doc_bin.add(doc)

    return doc_bin

def train_and_save_model(training_data, n_iter=30):
    """
    Train a custom NER model using DocBin training.
    """
    if os.path.exists(MODEL_DIR):
        print(f"Model directory '{MODEL_DIR}' exists. Loading existing model.")
        return spacy.load(MODEL_DIR)

    # Blank English model
    nlp = spacy.blank("en")

    # Add NER pipe
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Add labels
    for example in training_data:
        for start, end, label in example["entities"]:
            ner.add_label(label)

    # Create DocBin
    doc_bin = create_docbin(training_data)
    docs = list(doc_bin.get_docs(nlp.vocab))

    # Begin training
    optimizer = nlp.begin_training()

    for epoch in range(n_iter):
        losses = {}
        random.shuffle(docs)
        for doc in docs:
            example = spacy.training.Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]})
            nlp.update([example], sgd=optimizer, drop=0.2, losses=losses)
        print(f"Epoch {epoch+1}, Losses: {losses}")

    # Save model
    nlp.to_disk(MODEL_DIR)
    print(f"Model trained and saved at '{MODEL_DIR}'")
    return nlp

def load_model():
    """
    Load trained NER model or train new one using Corona2.json data.
    Returns:
        spacy.language.Language: Loaded or newly trained NER model
    Raises:
        ValueError: If Corona2.json cannot be loaded or is invalid
    """
    if os.path.exists(MODEL_DIR):
        print(f"Loading existing model from {MODEL_DIR}")
        return spacy.load(MODEL_DIR)
    
    try:
        with open("Corona2.json", "r", encoding="utf-8") as f:
            training_data = json.load(f)
            print(f"Successfully loaded {len(training_data)} training examples from Corona2.json")
            return train_and_save_model(training_data)
    except FileNotFoundError:
        raise ValueError("Corona2.json not found in current directory")
    except json.JSONDecodeError:
        raise ValueError("Corona2.json contains invalid JSON format")
    except Exception as e:

        raise ValueError(f"Error loading training data: {str(e)}")        
