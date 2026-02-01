import pandas as pd
import streamlit as st
import numpy as np
import torch
import sys
from pathlib import Path
from catboost import CatBoostClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.utils.preprocessor import get_tagged_context

# --- 1. CONFIGURATION & CONSTANTS ---

st.set_page_config(page_title="Propaganda Detector", page_icon="📢", layout="centered")

CATBOOST_REPO_ID = "hannusia123123/propaganda-stacking-catboost"
CATBOOST_FILENAME = "catboost.cbm"

TRANSFORMER_MODELS = {
    "roberta": "hannusia123123/propaganda-roberta-base",     
    "deberta": "hannusia123123/propaganda-deberta-v3-base", 
    "xlnet": "hannusia123123/propaganda-xlnet-base",         
    "toxic_bert": "hannusia123123/propaganda-toxic-bert" 
}

CLASS_WEIGHTS = np.array([
    0.8, 1.2, 0.8, 0.8, 0.9, 1.2, 1.6, 1.1, 3.2, 1.7, 3.3, 0.8, 0.8, 1.0
])

LABELS = [
    'Appeal_to_Authority', 'Appeal_to_fear-prejudice', 'Bandwagon,Reductio_ad_hitlerum',
    'Black-and-White_Fallacy', 'Causal_Oversimplification', 'Doubt',
    'Exaggeration,Minimisation', 'Flag-Waving', 'Loaded_Language',
    'Name_Calling,Labeling', 'Repetition', 'Slogans',
    'Thought-terminating_Cliches', 'Whataboutism,Straw_Men,Red_Herring'
]

THRESHOLDS = {
    "Appeal_to_Authority": 0.32,
    "Appeal_to_fear-prejudice": 0.33,
    "Bandwagon,Reductio_ad_hitlerum": 0.41,
    "Black-and-White_Fallacy": 0.26,
    "Causal_Oversimplification": 0.35,
    "Doubt": 0.35,
    "Exaggeration,Minimisation": 0.36,
    "Flag-Waving": 0.46,
    "Loaded_Language": 0.50,
    "Name_Calling,Labeling": 0.55,
    "Repetition": 0.51,
    "Slogans": 0.44,
    "Thought-terminating_Cliches": 0.25,
    "Whataboutism,Straw_Men,Red_Herring": 0.33,
    "default": 0.30
}

EXPLANATIONS = {
    "Appeal_to_Authority": "Citing an expert or authority figure to support a claim, even if they are not an expert in the specific field discussed.",
    "Appeal_to_fear-prejudice": "Building an argument on fear or prejudice rather than logic to scare the audience into accepting a specific conclusion.",
    "Bandwagon,Reductio_ad_hitlerum": "Attempting to persuade the audience to join in because 'everyone else is doing it', or associating the opponent with a despised figure (like Hitler).",
    "Black-and-White_Fallacy": "Presenting only two options (good vs. bad) as the only possibilities, ignoring any middle ground or nuance.",
    "Causal_Oversimplification": "Assuming a single, simple cause for a complex event when multiple causes are likely responsible.",
    "Doubt": "Questioning the credibility of someone or something without offering actual facts to support the suspicion.",
    "Exaggeration,Minimisation": "Making something seem much better, worse, or more important (exaggeration) or less important (minimisation) than it really is.",
    "Flag-Waving": "Appealing to patriotism or national identity to justify an action or idea ('doing it for the country').",
    "Loaded_Language": "Using words with strong positive or negative emotional connotations to influence the audience's reaction.",
    "Name_Calling,Labeling": "Using derogatory labels or insults to attack the opponent personally instead of addressing their arguments.",
    "Repetition": "Repeating the same message, word, or phrase over and over again so that the audience will eventually accept it.",
    "Slogans": "Using a short, catchy, and memorable phrase that bypasses critical thinking.",
    "Thought-terminating_Cliches": "Using common phrases or proverbs (e.g., 'it is what it is') to stop an argument or debate.",
    "Whataboutism,Straw_Men,Red_Herring": "Deflecting criticism by pointing to another's faults (Whataboutism), attacking a distorted version of the argument (Straw Man), or introducing irrelevant topics (Red Herring)."
}

# --- 2. MODEL LOADING ---

@st.cache_resource
def load_models():
    models = {}
    tokenizers = {}
    
    with st.spinner("Loading Transformer Models..."):
        for name, path in TRANSFORMER_MODELS.items():
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForSequenceClassification.from_pretrained(path)
                model.eval()
                tokenizers[name] = tokenizer
                models[name] = model
            except Exception as e:
                st.warning(f"⚠️ Could not load {name} from {path}. Error: {e}")

    catboost_model = CatBoostClassifier()
    try:
        model_path = hf_hub_download(repo_id=CATBOOST_REPO_ID, filename=CATBOOST_FILENAME)
        catboost_model.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to download CatBoost model. Error: {e}")
        return None, None, None

    return tokenizers, models, catboost_model

tokenizers, transformer_models, meta_model = load_models()

# --- 3. PREDICTION LOGIC ---

def get_transformer_logits(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.numpy()[0]

def predict_propaganda(text):
    if not meta_model or not transformer_models:
        return "Models not loaded correctly", 0.0, []

    all_logits = []
    model_order = ["roberta", "deberta", "xlnet", "toxic_bert"]
    
    for name in model_order:
        if name in transformer_models:
            logits = get_transformer_logits(text, tokenizers[name], transformer_models[name])
            all_logits.extend(logits)
        else:
            all_logits.extend([0.0] * 14)

    features = np.array(all_logits).reshape(1, -1)
    probs = meta_model.predict_proba(features)[0]
    
    weighted_probs = probs * CLASS_WEIGHTS
    
    pred_idx = np.argmax(weighted_probs)
    confidence = weighted_probs[pred_idx] / np.sum(weighted_probs)
    predicted_label = LABELS[pred_idx]
    
    return predicted_label, confidence, (weighted_probs / np.sum(weighted_probs))

# --- 4. UI LAYOUT ---

st.title("📢 Propaganda Technique Detector")
st.markdown("Analyze text fragments for manipulative rhetorical techniques using an Ensemble of Transformers.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Context")
    paragraph_input = st.text_area(
        "Paste the full paragraph here:", 
        height=200,
        placeholder="Example: The senator claimed that taxes would destroy the economy, ignoring the data..."
    )

with col2:
    st.subheader("2. Fragment")
    fragment_input = st.text_area(
        "Copy the specific fragment from the left:", 
        height=200,
        placeholder="Example: destroy the economy"
    )

if st.button("🔍 Analyze Fragment", type="primary", use_container_width=True):
    if not paragraph_input or not fragment_input:
        st.warning("⚠️ Please fill in both fields.")
    else:
        processed_text, _, error = get_tagged_context(
            paragraph_input.strip(), 
            fragment=fragment_input.strip()
        )
        
        if error:
            st.error(f"❌ {error}")
            st.info("💡 Make sure you copied the fragment EXACTLY as it appears in the text (including punctuation).")
        else:
            with st.spinner("AI models are analyzing rhetorical patterns..."):
                try:
                    label, conf, all_probs_array = predict_propaganda(processed_text)
                    
                    st.divider()
                    
                    required_threshold = THRESHOLDS.get(label, THRESHOLDS["default"])
                    
                    if conf < required_threshold:
                        st.warning("🤔 **No strong propaganda technique detected.**")
                        st.markdown(f"The model detected a weak signal for **{label}** ({conf:.1%}).")
                    else:
                        st.success(f"### 🚨 Detected: **{label}**")
                        st.progress(float(conf), text=f"Confidence: {conf:.1%}")

                    results_df = pd.DataFrame({
                        'Technique': LABELS,
                        'Confidence': all_probs_array
                    }).sort_values(by='Confidence', ascending=False)

                    with st.expander("📊 View full probability distribution"):
                        st.bar_chart(results_df.set_index('Technique'))
                        st.table(results_df.style.format({'Confidence': '{:.1%}'}))

                    if label in EXPLANATIONS and conf >= required_threshold:
                        st.info(f"**📚 Definition:** {EXPLANATIONS[label]}")

                except Exception as e:
                    st.error(f"System Error: {e}")