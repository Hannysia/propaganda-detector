import os
import json
import spacy
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from collections import defaultdict
import pandas as pd

load_dotenv()

# --- CONFIGURATION ---
HF_DATASET_REPO = "hannusia123123/propaganda-detector-dataset"
HF_TOKEN = os.getenv("HF_TOKEN")
VAL_SIZE = 0.15
RANDOM_SEED = 42

SPACY_MODEL = "en_core_web_sm" 

# --- PATHS SETUP ---
RAW_PATH = 'data/raw'
PROCESSED_PATH = 'data/processed'

os.makedirs(PROCESSED_PATH, exist_ok=True)

def load_spans(label_file_path):
    spans = []
    if not label_file_path.exists():
        return spans
    with open(label_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                spans.append((int(parts[1]), int(parts[2])))
    return spans

def check_overlap(sent_start, sent_end, spans):
    for span_start, span_end in spans:
        if max(sent_start, span_start) < min(sent_end, span_end):
            return 1
    return 0

def get_local_spans(sent_start, sent_end, global_spans):
    local_spans = []
    for span_start, span_end in global_spans:
        if max(sent_start, span_start) < min(sent_end, span_end):
            overlap_start = max(sent_start, span_start)
            overlap_end = min(sent_end, span_end)
            local_spans.append([overlap_start - sent_start, overlap_end - sent_start])
    return local_spans

def build_dataset(raw_data_path):
    all_data = []
    raw_path = Path(raw_data_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Directory not found: {raw_data_path}")

    txt_files = sorted(list(raw_path.glob("*.txt")))
    print(f"🔍 Found {len(txt_files)} articles in {raw_data_path}")

    nlp = spacy.load(SPACY_MODEL)

    for txt_file in tqdm(txt_files, desc="Parsing articles & merging sentences"):
        article_id = txt_file.stem
        label_file = raw_path / f"{article_id}.task1-SI.labels"
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                article_text = f.read()
                
            spans = load_spans(label_file)
            doc = nlp(article_text)
            
            # --- SMART MERGING LOGIC ---
            merged_sents = []
            for sent in doc.sents:
                text_slice = article_text[sent.start_char:sent.end_char]
                stripped_test = text_slice.strip()
                
                if not merged_sents:
                    merged_sents.append({
                        "start_char": sent.start_char, 
                        "end_char": sent.end_char, 
                        "text": text_slice
                    })
                    continue
                
                has_alnum = any(c.isalnum() for c in stripped_test)
                if len(stripped_test) < 5 or not has_alnum:
                    merged_sents[-1]["end_char"] = sent.end_char
                    merged_sents[-1]["text"] = article_text[merged_sents[-1]["start_char"]:sent.end_char]
                else:
                    merged_sents.append({
                        "start_char": sent.start_char, 
                        "end_char": sent.end_char, 
                        "text": text_slice
                    })

            # --- CONTEXT WINDOW GENERATION ---
            for idx, m_sent in enumerate(merged_sents):
                start_c = m_sent["start_char"]
                end_c = m_sent["end_char"]
                text = m_sent["text"]
                
                prev_text = merged_sents[idx - 1]["text"] if idx > 0 else ""
                next_text = merged_sents[idx + 1]["text"] if idx < len(merged_sents) - 1 else ""
                
                label = check_overlap(start_c, end_c, spans)
                local_spans = get_local_spans(start_c, end_c, spans)
                
                all_data.append({
                    "article_id": article_id,
                    "sentence_idx": idx,
                    "start_char": start_c,
                    "end_char": end_c,
                    "prev_text": prev_text,
                    "text": text,
                    "next_text": next_text,
                    "propaganda_spans": local_spans,
                    "label": label
                })

        except Exception as e:
            print(f"❌ Error processing {article_id}: {e}")

    # --- STATISTICS ---
    total = len(all_data)
    propaganda = sum(1 for d in all_data if d["label"] == 1)
    clean = total - propaganda
    
    print(f"\n📊 DATASET STATISTICS:")
    print(f"Total sentences: {total}")
    print(f"Propaganda (1): {propaganda} | Clean (0): {clean}")
    if total > 0:
        print(f"Balance: {propaganda/total:.2%}")

    return all_data

def process_dataset(data):
    articles = defaultdict(list)
    for item in data:
        articles[item["article_id"]].append(item)
        
    article_ids = list(articles.keys())
    article_densities = []
    
    for art_id in article_ids:
        sents = articles[art_id]
        prop_count = sum(1 for s in sents if s["label"] == 1)
        density = prop_count / len(sents) if len(sents) > 0 else 0
        article_densities.append(density)
        
    bins = [-1, 0.0001, 0.10, 0.25, 1.0]
    labels = [0, 1, 2, 3]
    categories = pd.cut(article_densities, bins=bins, labels=labels)
    
    train_ids, val_ids = train_test_split(
        article_ids, 
        test_size=VAL_SIZE, 
        random_state=RANDOM_SEED, 
        stratify=categories
    )
    
    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)
    
    train_data = [item for item in data if item["article_id"] in train_ids_set]
    val_data = [item for item in data if item["article_id"] in val_ids_set]
    
    train_prop = sum(1 for d in train_data if d["label"] == 1)
    val_prop = sum(1 for d in val_data if d["label"] == 1)
    
    print(f"✅ Article-level Split successful!")
    print(f"   Train: {len(train_data)} sentences (Propaganda ratio: {train_prop/len(train_data):.2%})")
    print(f"   Val:   {len(val_data)} sentences (Propaganda ratio: {val_prop/len(val_data):.2%})")
    
    return train_data, val_data

if __name__ == "__main__":
    print("🚀 Starting dataset generation process...")
    full_data = build_dataset(RAW_PATH)

    train_data, val_data = process_dataset(full_data)
    
    output_file = os.path.join(PROCESSED_PATH, 'si_dataset.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in full_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"\n✅ Local dataset saved to: {output_file}")
    
    if HF_TOKEN:
        hf_config_name = "si_sc_dataset"
        print(f"☁️ Uploading to Hugging Face: {HF_DATASET_REPO} (Config: {hf_config_name})...")
        try:
            dataset_dict = DatasetDict({
                'full': Dataset.from_list(full_data),
                'train': Dataset.from_list(train_data),
                'validation': Dataset.from_list(val_data)
            })
            
            dataset_dict.push_to_hub(
                HF_DATASET_REPO, 
                config_name=hf_config_name,
                token=HF_TOKEN,
                private=True
            )
            print("🎉 Dataset pushed to Hugging Face successfully in JSONL format!")
        except Exception as e:
            print(f"❌ Failed to upload to Hugging Face: {e}")
    else:
        print("⚠️ HF_TOKEN not found in environment variables. Upload skipped.")