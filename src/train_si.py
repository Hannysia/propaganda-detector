import os
import sys
import gc
import torch
import wandb
import numpy as np
from datetime import datetime
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback

sys.path.append(os.getcwd())

from src.models.span_detector import PropagandaSpanDetector
from src.models.trainer import SITrainer 
from src.data.dataset_si import PropagandaSIDataset 
from src.utils.common import seed_everything
from src.utils.metrics import compute_si_metrics

# --- 1. CONFIG & SETUP ---
HF_TOKEN = os.getenv("HF_TOKEN")

def run_si_pipeline(
    model_name: str,
    hf_model_repo: str,
    run_prefix: str,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    num_train_epochs: int = 5,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "cosine",
    gradient_accumulation_steps: int = 1,
    max_length: int = 512,
    merge_threshold: int = 0,
    focal_weight: float = 1.0,
    custom_dropout: float = 0.1,
    push_model_to_hub: bool = True,
    early_stopping_patience: int = 3,
    source_dataset_repo: str = "hannusia123123/propaganda-detector-dataset"
):
    # --- 1. SETUP ---
    GLOBAL_SEED = 42
    seed_everything(GLOBAL_SEED)
    api = HfApi()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_id = f"{run_prefix}-{timestamp}"
    print(f"🚀 STARTING PIPELINE: {run_id}")
    
    wandb.init(project="propaganda-detector-si", name=run_id, config=locals())

    # --- 2. DATA DOWNLOAD  ---
    print(f"☁️ Downloading preprocessed data from Hugging Face: {source_dataset_repo}")
    
    dataset = load_dataset(
        source_dataset_repo, 
        name="span_identification",
        token=HF_TOKEN,
        download_mode="force_redownload"
    )
    
    train_data = dataset['train']
    val_data = dataset['validation']

    print(f"📊 Data Loaded: Train={len(train_data)}, Val={len(val_data)}")

    # --- 3. TOKENIZER & DATASETS ---
    print("⚙️ Initializing Tokenizer & Datasets...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = PropagandaSIDataset(
        data=train_data,
        tokenizer=tokenizer, 
        max_length=max_length
    )
    
    val_dataset = PropagandaSIDataset(
        data=val_data,
        tokenizer=tokenizer, 
        max_length=max_length
    )

    # --- 4. POS_WEIGHT CALCULATION (For Trainer) ---
    print("⚖️ Calculating Pos Weight for Trainer...")
    counts = np.zeros(3)
    for item in train_dataset:
        labels = item['labels']
        counts[0] += (labels == 0).sum().item()
        counts[1] += (labels == 1).sum().item()
        counts[2] += (labels == 2).sum().item()
    
    pos_weight = counts[0] / (counts[1] + counts[2] + 1e-9)
    print(f"⚖️ Pos Weight set to: {pos_weight:.2f}")

    # --- 5. MODEL SETUP ---
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    config.focal_weight = focal_weight

    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = custom_dropout
    if hasattr(config, "dropout"):
        config.dropout = custom_dropout

    model = PropagandaSpanDetector(model_name=model_name, num_labels=3, config=config)
    model.focal_weight = focal_weight 
    model = model.float()

    for param in model.parameters():
        param.data = param.data.contiguous()

    # --- 6. TRAINING ARGUMENTS ---
    training_args = TrainingArguments(
        output_dir=f"./results_si/{run_id}",
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
        metric_for_best_model="f1_symbolic", 
        greater_is_better=True,
        save_total_limit=1,
        fp16=False,
        bf16=False,
        push_to_hub=False 
    )

    # --- 7. TRAINER INITIALIZATION ---    
    def compute_metrics_wrapper(eval_pred):
    
        return compute_si_metrics(
            eval_preds=eval_pred, 
            eval_dataset=val_dataset,       
            merge_threshold=merge_threshold 
        )

    trainer = SITrainer(
        pos_weight=pos_weight, 
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_wrapper, 
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )

    trainer.use_amp = False

    # --- 8. TRAINING LOOP ---
    print("🚀 Starting training...")
    trainer.train()

    # --- 9. SAVING BEST MODEL TO HF ---
    if push_model_to_hub:
        print(f"🔥 Training finished. Uploading best {run_prefix} model to HF...")
        
        best_model_path = f"./best_model_si/{run_prefix}"
        
        trainer.save_model(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        
        try:
            api.create_repo(repo_id=hf_model_repo, exist_ok=True)
            api.upload_folder(
                folder_path=best_model_path,
                repo_id=hf_model_repo,
                path_in_repo=run_prefix,
                commit_message=f"Add best {run_prefix} model from run {run_id}"
            )
            print(f"✅ Successfully uploaded to HF: https://huggingface.co/{hf_model_repo}/tree/main/{run_prefix}")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
    else:
        print("🛑 Sweep mode: skipping model save and Hugging Face upload.")

    # --- 10. CLEANUP ---
    wandb.finish()
    del model, trainer, train_dataset, val_dataset
    torch.cuda.empty_cache()
    gc.collect()
    
    print("🏁 Pipeline Finished.")