import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from datasets import Dataset
from dotenv import load_dotenv
from .preprocessor import get_tagged_context
load_dotenv()

# --- CONFIGURATION ---
HF_DATASET_REPO = "hannusia123123/propaganda-detector-dataset"
HF_TOKEN = os.getenv("HF_TOKEN")
N_FOLDS = 5
HOLDOUT_SIZE = 0.1 
SEARCH_SEED_RANGE = range(0, 1000)

# --- PATHS SETUP ---
RAW_PATH = 'data/raw' 
PROCESSED_PATH = 'data/processed'

os.makedirs(PROCESSED_PATH, exist_ok=True)

def build_dataset(raw_data_path):
    all_data = []
    
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Directory not found: {raw_data_path}")

    label_files = [f for f in os.listdir(raw_data_path) if f.endswith('.task2-TC.labels')]
    
    print(f"Found {len(label_files)} label files in {raw_data_path}")
        
    for label_file in tqdm(label_files, desc="Articles processing"):
        article_id = label_file.split('.')[0]
        txt_full_path = os.path.join(raw_data_path, f"{article_id}.txt")
        labels_full_path = os.path.join(raw_data_path, label_file)
        
        if not os.path.exists(txt_full_path):
            continue

        with open(txt_full_path, 'r', encoding='utf-8') as f:
            article_text = f.read()

        df_labels = pd.read_csv(labels_full_path, sep='\t', names=['article_id', 'technique', 'start', 'end'])
        
        for _, row in df_labels.iterrows():
            context_clean, fragment_clean, error = get_tagged_context(
                text=article_text, 
                start_char=int(row['start']), 
                end_char=int(row['end'])
            )
            
            if error:
                continue

            record = {
                'article_id': article_id,
                'start_offset': int(row['start']),
                'fragment': fragment_clean,
                'context': context_clean,
                'label': row['technique']
            }
            
            if record not in all_data:
                all_data.append(record)           
            
    return pd.DataFrame(all_data)


def find_best_split_seed(df, n_splits, group_col='article_id', target_col='label', seed_range=range(200)):

    print(f"🔍 Searching for ideal seed in range {seed_range}...")
    
    best_seed = 0
    min_error = float('inf')
    
    global_dist = df[target_col].value_counts(normalize=True).sort_index()

    for seed in tqdm(seed_range, desc="Seed Search"):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        try:
            train_idx, val_idx = next(sgkf.split(df, df[target_col], groups=df[group_col]))
            
            val_df = df.iloc[val_idx]
            val_dist = val_df[target_col].value_counts(normalize=True).sort_index()
            
            combined_dist = pd.DataFrame({'global': global_dist, 'val': val_dist}).fillna(0)
            
            error = ((combined_dist['global'] - combined_dist['val']) ** 2).sum()
            
            if error < min_error:
                min_error = error
                best_seed = seed
        except:
            continue
            
    print(f"Best Seed Found: {best_seed} (Error: {min_error:.6f})")
    return best_seed


def process_dataset(df):
    print("Starting dataset splitting...")
    
    n_splits_holdout = int(1 / HOLDOUT_SIZE)
    best_holdout_seed = find_best_split_seed(
        df, 
        n_splits=n_splits_holdout, 
        seed_range=SEARCH_SEED_RANGE
    )
    
    sgkf_holdout = StratifiedGroupKFold(n_splits=n_splits_holdout, shuffle=True, random_state=best_holdout_seed)
    main_idx, holdout_idx = next(sgkf_holdout.split(df, df['label'], groups=df['article_id']))
    
    df['is_holdout'] = False
    df['fold'] = -1
    
    df.loc[holdout_idx, 'is_holdout'] = True
    
    print(f"✅ Holdout separated. Main: {len(main_idx)}, Holdout: {len(holdout_idx)}")
    
    df_main = df[df['is_holdout'] == False].reset_index(drop=True)
    original_indices = df.index[df['is_holdout'] == False]
    
    cv_seed = 42 
    sgkf_cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=cv_seed)
    
    for fold, (train_i, val_i) in enumerate(sgkf_cv.split(df_main, df_main['label'], groups=df_main['article_id'])):
        val_original_idx = original_indices[val_i]
        df.loc[val_original_idx, 'fold'] = fold
        
    print("✅ Folds assigned (0-4) to Main data.")
    return df


if __name__ == "__main__":
    final_df = build_dataset(RAW_PATH)
    print(f"Total raw examples: {len(final_df)}")

    enriched_df = process_dataset(final_df)
    
    output_file = os.path.join(PROCESSED_PATH, 'dataset.csv')
    enriched_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✅ Local dataset saved to: {output_file}")
    
    if HF_TOKEN:
        print(f"☁️ Uploading to Hugging Face: {HF_DATASET_REPO}...")
        try:
            hf_dataset = Dataset.from_pandas(enriched_df)
            
            hf_dataset.push_to_hub(HF_DATASET_REPO, token=HF_TOKEN)
            print("🎉 SUCCESS! Dataset is live on Hugging Face.")
            print(f"🔗 Link: https://huggingface.co/datasets/{HF_DATASET_REPO}")
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
    else:
        print("⚠️ HF_TOKEN not found. Skipping upload.")
