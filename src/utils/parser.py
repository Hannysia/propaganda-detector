import spacy
import os
import pandas as pd
from tqdm import tqdm

if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    RAW_PATH = '/kaggle/input/semeval-2020-task-11/data/raw'
    PROCESSED_PATH = '/kaggle/working'
else:
    RAW_PATH = 'data/raw' 
    PROCESSED_PATH = 'data/processed'

os.makedirs(PROCESSED_PATH, exist_ok=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def build_dataset(raw_data_path):
    all_data = []
    
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Directory not found: {raw_data_path}")

    label_files = [f for f in os.listdir(raw_data_path) if f.endswith('.task2-TC.labels')]
    
    print(f"Found {len(label_files)} label files in {raw_data_path}")
        
    for label_file in tqdm(label_files, desc="Articles processing"):
        article_id = label_file.split('.')[0]
        txt_file = f"{article_id}.txt"
        
        txt_full_path = os.path.join(raw_data_path, txt_file)
        labels_full_path = os.path.join(raw_data_path, label_file)
        
        if not os.path.exists(txt_full_path):
            continue

        with open(txt_full_path, 'r', encoding='utf-8') as f:
            article_text = f.read()

        doc = nlp(article_text)
        sentences = list(doc.sents)
            
        df_labels = pd.read_csv(labels_full_path, sep='\t', names=['article_id', 'technique', 'start', 'end'])
        
        for _, row in df_labels.iterrows():
            start, end = int(row['start']), int(row['end'])
            fragment = article_text[start:end]
            
            context = f" <E> {fragment} </E> "
            
            for sent in sentences:
                if sent.start_char <= start < sent.end_char:
                    rel_start = start - sent.start_char
                    rel_end = end - sent.start_char
                    
                    context = (
                        sent.text[:rel_start] + 
                        " <E> " + 
                        sent.text[rel_start:rel_end] + 
                        " </E> " + 
                        sent.text[rel_end:]
                    )
                    context = context.replace('\n', ' ').strip()
                    break

            record = {
                'article_id': article_id,
                'fragment': fragment,
                'context': context,
                'label': row['technique']
            }
            if record not in all_data:
                all_data.append(record)           
            
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    final_df = build_dataset(RAW_PATH)

    output_file = os.path.join(PROCESSED_PATH, 'dataset.csv')
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n✅ Done! Dataset saved to: {output_file}")
    print(f"Total examples: {len(final_df)}")