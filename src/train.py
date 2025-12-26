import os
import sys
import pandas as pd
import numpy as np
import torch
import wandb
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments
)
from sklearn.model_selection import StratifiedGroupKFold

sys.path.append(os.getcwd())
from src.data import PropagandaDataset
from src.models import WeightedLossTrainer
from src.utils import (
    setup_environment, 
    print_distribution, 
    compute_metrics, 
    log_confusion_matrix
)


# --- 1. CONFIG & SETUP ---
DATA_PATH, HF_TOKEN = setup_environment()

MODEL_NAME = "bert-base-uncased"
# У W&B це буде назва конкретного експерименту (з часом запуску)
RUN_NAME = f"span-bert-{datetime.now().strftime('%d-%m-%H-%M')}"

# Ця змінна лишається, щоб ми знали, куди пушити, ЯКЩО захочемо
HF_REPO_NAME = "hannusia123123/propaganda-technique-detector"


# --- 2. PREPARE DATA ---
print("📊 Loading and splitting data...")
df = pd.read_csv(DATA_PATH)

labels_list = sorted(df['label'].unique())
label2id = {label: i for i, label in enumerate(labels_list)}
id2label = {i: label for label, i in label2id.items()}

print("✂️ Doing Stratified Group Split...")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_index, val_index = next(sgkf.split(df['context'], df['label'], groups=df['article_id']))
train_df = df.iloc[train_index]
val_df = df.iloc[val_index]

print("-" * 40)

print_distribution(train_df, "TRAIN")
print_distribution(val_df, "VALIDATION")

print("-" * 40)


# --- 3. DATASETS & WEIGHTS ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

special_tokens_dict = {'additional_special_tokens': ['<E>', '</E>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

train_dataset = PropagandaDataset(train_df['context'].tolist(), [label2id[l] for l in train_df['label']], tokenizer)
val_dataset = PropagandaDataset(val_df['context'].tolist(), [label2id[l] for l in val_df['label']], tokenizer)

print("⚖️ Calculating Class Weights...")
train_labels = [x['labels'].item() for x in train_dataset]
class_weights_arr = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = torch.tensor(class_weights_arr, dtype=torch.float).to(device)


# --- 4. MODEL & TRAINING ---
wandb.init(project="propaganda-detector", name=RUN_NAME, job_type="train")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(labels_list),
    id2label=id2label,
    label2id=label2id
)

model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    save_total_limit=2,
    push_to_hub=False,   
)

trainer = WeightedLossTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer 
)

print("🚀 Starting training...")
trainer.train()

log_confusion_matrix(trainer, val_dataset, id2label)

wandb.finish()
print("✅ Done!")