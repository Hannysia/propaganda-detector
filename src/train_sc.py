import os
import sys
import gc
import torch
import wandb
import numpy as np
from datetime import datetime
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from src.models.trainer import WeightedLossTrainer
import torch.nn as nn

sys.path.append(os.getcwd())
from src.utils.common import seed_everything

# --- 1. CONFIG & SETUP ---
HF_TOKEN = os.getenv("HF_TOKEN")

def run_sc_pipeline(
    model_name: str = "microsoft/deberta-v3-base",
    hf_model_repo: str = "hannusia123123/deberta-sentence-classifier",
    run_prefix: str = "sc-deberta",
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    num_train_epochs: int = 4,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "cosine",
    max_length: int = 128,
    custom_dropout: float = 0.1,
    push_model_to_hub: bool = True,
    early_stopping_patience: int = 2,
    source_dataset_repo: str = "hannusia123123/propaganda-detector-dataset"
):
    # --- 1. SETUP ---
    GLOBAL_SEED = 42
    seed_everything(GLOBAL_SEED)
    api = HfApi()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_id = f"{run_prefix}-{timestamp}"
    print(f"🚀 STARTING SC PIPELINE: {run_id}")
    
    wandb.init(project="propaganda-detector-sc", name=run_id, config=locals())

    # --- 2. DATA DOWNLOAD  ---
    print(f"☁️ Downloading preprocessed data from Hugging Face: {source_dataset_repo}")
    
    dataset = load_dataset(
        source_dataset_repo, 
        name="sentence_classification",
        token=HF_TOKEN,
        download_mode="force_redownload"
    )
    
    train_data = dataset['train']
    val_data = dataset['validation']
    print(f"📊 Data Loaded: Train={len(train_data)}, Val={len(val_data)}")

    # --- 3. TOKENIZER & PREPROCESSING ---
    print("⚙️ Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    train_dataset = train_data.map(tokenize_fn, batched=True)
    val_dataset = val_data.map(tokenize_fn, batched=True)

    # --- 4. CLASS WEIGHTS CALCULATION ---
    print("⚖️ Calculating Class Weights for Trainer...")
    labels = train_dataset['label']
    counts = np.bincount(labels)
    weight_0 = 1.0
    weight_1 = counts[0] / (counts[1] + 1e-9)
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float)
    print(f"⚖️ Class Weights: Clean(0)={weight_0:.2f}, Propaganda(1)={weight_1:.2f}")

    # --- 5. MODEL SETUP ---
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    
    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = custom_dropout
    if hasattr(config, "attention_probs_dropout_prob"):
        config.attention_probs_dropout_prob = custom_dropout

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # --- 6. TRAINING ARGUMENTS ---
    training_args = TrainingArguments(
        output_dir=f"./results_sc/{run_id}",
        run_name=run_id,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2, 
        optim="adamw_torch",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=1.0,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=True,
        push_to_hub=False 
    )

    # --- 7. TRAINER INITIALIZATION ---    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        }

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )

    # --- 8. TRAINING LOOP ---
    print("🚀 Starting training...")
    trainer.train()

    # --- 9. WANDB CUSTOM PLOTS (PR Curve & Confusion Matrix) ---
    print("📊 Generating custom W&B evaluation plots...")
    eval_results = trainer.predict(val_dataset)
    logits = eval_results.predictions
    y_true = eval_results.label_ids
    
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    y_preds = np.argmax(probs, axis=1)
    
    wandb.log({"pr_curve": wandb.plot.pr_curve(y_true, probs, labels=["Clean", "Propaganda"])})
    
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(
        probs=None, 
        y_true=y_true, 
        preds=y_preds, 
        class_names=["Clean", "Propaganda"]
    )})

    # --- 10. SAVING BEST MODEL TO HF ---
    if push_model_to_hub:
        print(f"🔥 Training finished. Uploading best {run_prefix} model to HF...")
        best_model_path = f"./best_model_sc/{run_prefix}"
        trainer.save_model(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        
        try:
            api.create_repo(repo_id=hf_model_repo, exist_ok=True)
            api.upload_folder(
                folder_path=best_model_path,
                repo_id=hf_model_repo,
                commit_message=f"Add best {run_prefix} model from run {run_id}"
            )
            print(f"✅ Successfully uploaded to HF: https://huggingface.co/{hf_model_repo}")
        except Exception as e:
            print(f"❌ Upload failed: {e}")

    # --- 11. CLEANUP ---
    wandb.finish()
    del model, trainer, train_dataset, val_dataset
    torch.cuda.empty_cache()
    gc.collect()
    
    print("🏁 SC Pipeline Finished.")

if __name__ == "__main__":
    run_sc_pipeline()