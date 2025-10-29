## CuisiNER: Filipino Food NER (spaCy + Streamlit) by caecht

Identify Filipino food entities (e.g., Sinigang, Adobo, Lechon) in text using a lightweight spaCy pipeline and an interactive Streamlit UI.

### Features
- Recognizes dozens of Filipino dishes via a custom `EntityRuler` label `FILIPINO_FOOD`
- Works alongside standard spaCy entities (PERSON, ORG, GPE, etc.)
- Interactive web app for entering text, inspecting entities, token details, and visualizations
- Simple evaluation script with confusion matrix and metrics visualization

### Project Structure
- `app.py`: Streamlit app for CuisiNER (primary UI)
- `filipino_food_config.py`: Food lists, variations, display config, and `FilipinoFoodNER` helper
- `ner_evaluation.py`: Simple evaluator that generates metrics and a visualization PNG
- `filipinoNer.py`: Minimal demo Streamlit app using a different (Tagalog) spaCy model
- `requirements.txt`: Minimal dependencies
- Images/outputs: `simple_filipino_food_ner_evaluation.png`, `filipino_food_test_results.csv`, other PNGs

### Prerequisites
- Python 3.9+ recommended
- pip

You also need the small English spaCy model used as a base in `filipino_food_config.py`:

```bash
python -m spacy download en_core_web_sm
```

If you plan to try the minimal demo in `filipinoNer.py`, ensure the Tagalog model it references is available or adjust that file accordingly.

### Installation
```bash
# from the project root
python -m venv .venv
. .venv/Scripts/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run the App
```bash
streamlit run app.py
```

Then open the local URL shown in your terminal. Enter sample text and click “Analyze Text” to see:
- All entities grouped by label (Filipino foods first)
- Detailed token table (expandable)
- Entity visualization (spaCy displaCy)

### Evaluation
The evaluator in `ner_evaluation.py` loads the CuisiNER model and runs a simple test suite of Filipino food sentences versus non-food sentences, printing metrics and saving a visualization.

```bash
python ner_evaluation.py
```

Outputs include:
- `simple_filipino_food_ner_evaluation.png`: confusion matrix, per-class scores, overall accuracy

Note: The test is synthetic and rule-based; it’s useful for sanity checks, not as a rigorous benchmark.

### How It Works (High-level)
- `FilipinoFoodNER.load_model_with_ruler()` loads `en_core_web_sm`, adds an `EntityRuler`, and injects patterns for items in `FILIPINO_FOODS` plus variations in `FILIPINO_FOOD_VARIATIONS`, all labeled as `FILIPINO_FOOD`.
- The Streamlit app uses that pipeline to process user text and render results and visualizations.

### Trying the Minimal Demo (optional)
`filipinoNer.py` demonstrates a simpler app that loads a Tagalog model (`tl_calamancy_md-0.1.0`). If you don’t have that model, either install it or switch it to a model you have installed.

### Troubleshooting
- If you get a spaCy model error: ensure `en_core_web_sm` is installed via `python -m spacy download en_core_web_sm`.
- If Streamlit won’t start: confirm the virtual environment is activated and dependencies are installed.
- On Windows PowerShell, activate with: `./.venv/Scripts/Activate.ps1` (adjust path if your venv name differs).

### License
MIT (or update accordingly)

### Acknowledgments
- Built with spaCy and Streamlit. Inspired by Filipino cuisine and domain-specific NLP customization.


