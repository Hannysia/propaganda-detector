import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
HF_DATASET_REPO = "hannusia123123/propaganda-detector-dataset"
HF_TOKEN = os.getenv("HF_TOKEN")
VAL_SIZE = 0.15
RANDOM_SEED = 42

# --- PATHS SETUP ---
RAW_PATH = 'data/raw'
PROCESSED_PATH = 'data/processed'

os.makedirs(PROCESSED_PATH, exist_ok=True)

def build_dataset(raw_data_path):
    all_data = []

    raw_path = Path(raw_data_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Directory not found: {raw_data_path}")

    txt_files = sorted(list(raw_path.glob("*.txt")))
    
    print(f"🔍 Found {len(txt_files)} articles in {raw_data_path}")

    for txt_file in tqdm(txt_files, desc="Parsing articles"):
        article_id = txt_file.stem
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                article_text = f.read()
        except Exception:
            continue

        label_file = raw_path / f"{article_id}.task1-SI.labels"
        spans = []
        
        if label_file.exists():
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            try:
                                start_idx = int(parts[1])
                                end_idx = int(parts[2])
                                spans.append({
                                    'start': start_idx, 
                                    'end': end_idx,
                                    'fragment': article_text[start_idx:end_idx]
                                })
                            except ValueError:
                                continue
            except Exception:
                pass

        all_data.append({
            'article_id': article_id,
            'text': article_text,
            'spans': spans 
        })
        
    return all_data


def process_dataset(all_data):
    print("⚙️ Starting dataset splitting...")
    
    article_densities = []
    for item in all_data:
        text_len = len(item['text'])
        if text_len == 0:
            article_densities.append(0)
            continue
        prop_chars = sum([span['end'] - span['start'] for span in item['spans']])
        article_densities.append(prop_chars / text_len)

    bins = [-1, 0.0001, 0.10, 0.25, 1.0]
    labels = [0, 1, 2, 3]
    categories = pd.cut(article_densities, bins=bins, labels=labels)

    train_data, val_data = train_test_split(
        all_data, 
        test_size=VAL_SIZE, 
        random_state=RANDOM_SEED, 
        stratify=categories
    )
    
    print(f"✅ Split successful. Train: {len(train_data)}, Val: {len(val_data)}")
    return train_data, val_data


if __name__ == "__main__":
    full_data = build_dataset(RAW_PATH)
    print(f"Total raw examples: {len(full_data)}")

    train_data, val_data = process_dataset(full_data)
    
    output_file = os.path.join(PROCESSED_PATH, 'si_dataset.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Local dataset saved to: {output_file}")
    
    if HF_TOKEN:
        print(f"☁️ Uploading to Hugging Face: {HF_DATASET_REPO} (Config: span_identification)...")
        try:
            dataset_dict = DatasetDict({
                'full': Dataset.from_list(full_data),
                'train': Dataset.from_list(train_data),
                'validation': Dataset.from_list(val_data)
            })
            
            dataset_dict.push_to_hub(
                HF_DATASET_REPO, 
                config_name="span_identification",
                token=HF_TOKEN
            )
            print("🎉 SUCCESS! SI Dataset is live on Hugging Face.")
            print(f"🔗 Link: https://huggingface.co/datasets/{HF_DATASET_REPO}")
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
    else:
        print("⚠️ HF_TOKEN not found. Skipping upload.")