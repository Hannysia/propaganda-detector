import re
import spacy
from tqdm import tqdm
import pandas as pd

print(f"✅ Loading Spacy model: en_core_web_trf ...")
NLP = spacy.load("en_core_web_trf")

def clean_punctuation(text):

    if not isinstance(text, str): return str(text)
    
    text = text.replace('“', '"').replace('”', '"').replace("’", "'").replace("‘", "'")
    text = text.replace('«', '"').replace('»', '"')
    
    text = text.replace('—', '-')
    
    return text

def mask_entities(text):

    doc = NLP(text)
    new_tokens = []
    
    for token in doc:
        if token.text in ["<", "E", ">", "/E", "</", "E>"]:
            new_tokens.append(token.text)
            continue
            
        if token.ent_type_:
            if token.ent_type_ == "PERSON":
                new_tokens.append("[PERSON]")
            elif token.ent_type_ == "ORG":
                new_tokens.append("[ORG]")
            elif token.ent_type_ in ["GPE", "LOC"]:
                new_tokens.append("[LOC]")
            elif token.ent_type_ == "NORP":
                new_tokens.append("[GROUP]")
            else:
                new_tokens.append(token.text)
        else:
            new_tokens.append(token.text)
            
    text = " ".join(new_tokens)
    
    text = text.replace("< E >", "<E>").replace("</ E >", "</E>").replace("< E>", "<E>").replace("</ E>", "</E>")
    
    text = re.sub(r'(\[PERSON\]\s*)+', '[PERSON] ', text)
    text = re.sub(r'(\[ORG\]\s*)+', '[ORG] ', text)
    text = re.sub(r'(\[LOC\]\s*)+', '[LOC] ', text)
    text = re.sub(r'(\[GROUP\]\s*)+', '[GROUP] ', text)
    
    text = re.sub(r'\s+([?.!,:;])', r'\1', text)
    
    return text.strip()

def apply_preprocessing(df):

    print("🚀 Preprocessor started...")
    
    df['context'] = df['context'].apply(clean_punctuation)
    df['fragment'] = df['fragment'].apply(clean_punctuation)

    tqdm.pandas(desc="Masking Entities")
    df['context'] = df['context'].progress_apply(mask_entities)
    
    print("✅ Preprocessing complete.")
    return df