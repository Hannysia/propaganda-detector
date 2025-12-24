import os
import pandas as pd
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datetime import datetime

load_dotenv() 

hf_token = None
wandb_key = None

if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    print("Detected Kaggle Environment ☁️")
    DATA_PATH = "/kaggle/working/dataset.csv"
    
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    
    # Hugging Face Login
    try:
        hf_token = user_secrets.get_secret("HF_TOKEN")
        from huggingface_hub import login
        login(token=hf_token)
        print("✅ Logged in to Hugging Face Hub")
    except:
        print("⚠️ HF_TOKEN not found. Model will NOT be pushed.")

    # W&B Login
    try:
        wandb_key = user_secrets.get_secret("WANDB_API_KEY")
        wandb.login(key=wandb_key)
        print("✅ Logged in to W&B")
    except:
        print("⚠️ WANDB_API_KEY not found.")
        
else:
    print("Detected Local Environment 🏠")
    DATA_PATH = "data/processed/dataset.csv"
    
    hf_token = os.getenv("HF_TOKEN")
    wandb.login()


MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128 
now = datetime.now().strftime("%d-%m-%H-%M")
RUN_NAME = f"baseline-bert-{now}"

wandb.init(
    project="propaganda-detector",
    job_type="baseline-training",
    name=RUN_NAME
)

df = pd.read_csv(DATA_PATH)

labels = sorted(df['label'].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

class PropagandaDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    f1_macro = f1_score(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro
    }

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

unique_article_ids = df['article_id'].unique()
np.random.seed(42)
np.random.shuffle(unique_article_ids)

train_size = int(len(unique_article_ids) * 0.8)
train_article_ids = unique_article_ids[:train_size]
val_article_ids = unique_article_ids[train_size:]

train_df = df[df['article_id'].isin(train_article_ids)]
val_df = df[df['article_id'].isin(val_article_ids)]

print(f"Train samples: {len(train_df)} (from {len(train_article_ids)} articles)")
print(f"Val samples: {len(val_df)} (from {len(val_article_ids)} articles)")

train_dataset = PropagandaDataset(
    train_df['context'].tolist(), 
    [label2id[l] for l in train_df['label']], 
    tokenizer
)
val_dataset = PropagandaDataset(
    val_df['context'].tolist(), 
    [label2id[l] for l in val_df['label']], 
    tokenizer
)

HF_REPO_NAME = "hannusia123123/propaganda-baseline-bert"  

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5, 
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    save_total_limit=2,
    
    push_to_hub=True,   
    hub_model_id=HF_REPO_NAME,        
    hub_token=hf_token,
    hub_strategy="every_save"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("🚀 Starting training...")
trainer.train()


save_path = "./best_baseline_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

artifact = wandb.Artifact(name=f"model-{RUN_NAME}", type="model")
artifact.add_dir(save_path)
wandb.log_artifact(artifact)

wandb.finish()

print("⬆️ Pushing to Hugging Face Hub...")
trainer.push_to_hub()
print("✅ Done!")