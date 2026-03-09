import argparse
import os
import sys
import gc
import pickle
import torch
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback, AutoConfig
import torch.nn.functional as F
from huggingface_hub import HfApi
import wandb

sys.path.append(os.getcwd())

from src.models.span_detector import PropagandaSpanDetector
from src.models.trainer import SITrainer 
from src.data.dataset_si import PropagandaSIDataset 
from src.utils.common import seed_everything
from src.utils.metrics import compute_si_metrics

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN") 
SOURCE_DATASET = "hannusia123123/propaganda-detector-dataset"
OOF_REPO_ID = "hannusia123123/si-oof"
MAX_LENGTH = 512
STRIDE = 256
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2

def get_model_config(model_name):
    if model_name == "roberta":
        return {
            "hf_path": "roberta-base",
            "lr": 2e-5,
            "focal_weight": 500.0,
            "weight_decay": 0.01,
            "dropout": 0.1,
            "batch_size": 16,
            "warmup_ratio": 0.1,
            "scheduler": "cosine"
        }
    elif model_name == "electra":
        return {
            "hf_path": "google/electra-base-discriminator",
            "lr": 1e-4,
            "focal_weight": 10.0,
            "weight_decay": 0.01,
            "dropout": 0.1,
            "batch_size": 16,
            "warmup_ratio": 0.1,
            "scheduler": "linear"
        }
    else:
        raise ValueError("Unknown model name")

def run_cv(model_name):
    seed_everything(42)
    cfg = get_model_config(model_name)
    print(f"🚀 Starting 5-Fold CV for: {model_name.upper()} ({cfg['hf_path']})")

    api = HfApi()
    
    try:
        api.create_repo(repo_id=OOF_REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
        print(f"☁️ Temporary repository for OOF backups is ready: {OOF_REPO_ID}")
    except Exception as e:
        print(f"⚠️ Failed to create repository (it might already exist): {e}")

    print("☁️ Loading the full dataset...")
    dataset = load_dataset(SOURCE_DATASET, name="span_identification", token=HF_TOKEN, split="full")
    all_data = list(dataset)
    print(f"📊 Total unique articles loaded: {len(all_data)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_path"])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = [] 

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
        print(f"\n" + "="*40)
        print(f"FOLD {fold+1} / 5")
        print("="*40)

        wandb.init(project="propaganda-detector-cv", name=f"{model_name}_fold{fold+1}", reinit=True)

        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]

        train_dataset = PropagandaSIDataset(train_data, tokenizer, MAX_LENGTH, STRIDE)
        val_dataset = PropagandaSIDataset(val_data, tokenizer, MAX_LENGTH, STRIDE)

        all_labels = [label for item in train_dataset for label in item['labels'] if label != -100]
        counts = np.bincount(all_labels)
        pos_weight = counts[0] / (counts[1] + counts[2] + 1e-9)

        config = AutoConfig.from_pretrained(cfg["hf_path"])
        config.focal_weight = cfg["focal_weight"]
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = cfg["dropout"]
        if hasattr(config, "dropout"):
            config.dropout = cfg["dropout"]

        model = PropagandaSpanDetector(model_name=cfg["hf_path"], num_labels=3, config=config)
        model.focal_weight = cfg["focal_weight"]
        model = model.float()

        training_args = TrainingArguments(
            output_dir=f"./cv_results/{model_name}_fold{fold+1}",
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"] * 2,
            optim="adamw_torch",
            learning_rate=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            lr_scheduler_type=cfg["scheduler"],
            warmup_ratio=cfg["warmup_ratio"],
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1_symbolic",
            greater_is_better=True,
            save_total_limit=1,
            fp16=False,
            report_to="wandb" 
        )

        def compute_metrics_wrapper(eval_pred):
            return compute_si_metrics(eval_preds=eval_pred, eval_dataset=val_dataset, merge_threshold=0)

        trainer = SITrainer(
            pos_weight=pos_weight,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_wrapper,
            processing_class=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
        )
        trainer.use_amp = False

        print(f"▶️ Training fold {fold+1}...")
        trainer.train()

        print(f"🔍 Generating OOF predictions for fold {fold+1}...")
        predictions = trainer.predict(val_dataset)
        logits = predictions.predictions 
        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()

        for i, window in enumerate(val_dataset.all_windows):
            oof_predictions.append({
                "article_id": window["article_id"],
                "probs": probs[i], 
                "offset_mapping": window["offset_mapping"],
                "attention_mask": window["attention_mask"],
                "labels": window["labels"] 
            })

        output_file = f"oof_{model_name}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(oof_predictions, f)
        
        print(f"⬆️ Uploading backup {output_file} to Hugging Face...")
        try:
            api.upload_file(
                path_or_fileobj=output_file,
                path_in_repo=output_file,
                repo_id=OOF_REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
                commit_message=f"Update {model_name} OOF after fold {fold+1}"
            )
            print("✅ Backup successfully saved to HF!")
        except Exception as e:
            print(f"❌ Backup upload failed: {e}")

        wandb.finish()

        del model, trainer, train_dataset, val_dataset
        torch.cuda.empty_cache()
        gc.collect()

    print(f"🎉 All 5 folds completed! Final OOF predictions saved in {OOF_REPO_ID}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["roberta", "electra"])
    args = parser.parse_args()
    run_cv(args.model)