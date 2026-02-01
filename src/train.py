import os
import sys
import gc
import pandas as pd
import numpy as np
import torch
import wandb
import shutil
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments,
    EarlyStoppingCallback
)
from huggingface_hub import HfApi

sys.path.append(os.getcwd())
from src.data import PropagandaDataset
from src.models import WeightedLossTrainer
from src.utils import (
    seed_everything,
    compute_metrics,
    print_distribution,
    log_confusion_matrix
)


# --- 1. CONFIG & SETUP ---
HF_TOKEN = os.getenv("HF_TOKEN")

def run_cv_pipeline(
    model_name: str,
    hf_model_repo: str,
    hf_oof_dataset_repo: str,
    run_prefix: str,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    num_train_epochs: int = 10,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 1,
    max_length: int = 196,
    n_folds: int = 5,
    source_dataset_repo: str = "hannusia123123/propaganda-detector-dataset"
):
    
    GLOBAL_SEED = 42
    seed_everything(GLOBAL_SEED)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_id = f"{run_prefix}-{timestamp}"
    
    print(f"🚀 STARTING PIPELINE: {run_id}")
        
    print(f"☁️ Downloading data from Hugging Face: {source_dataset_repo}")
    dataset = load_dataset(source_dataset_repo, token=HF_TOKEN)
    df = dataset['train'].to_pandas()
    
    labels_list = sorted(df['label'].unique())
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}

    df_holdout = df[df['is_holdout'] == True].reset_index(drop=True)
    df_main = df[df['is_holdout'] == False].reset_index(drop=True)

    print(f"📊 Data Split (Loaded from file): Main={len(df_main)}, Holdout={len(df_holdout)}")
    print_distribution(df_holdout, "HOLDOUT SET")
    print_distribution(df_main, "MAIN SET")

    oof_preds_list = []
    holdout_logits_sum = np.zeros((len(df_holdout), len(labels_list)))
    
    best_f1_global = 0

    # --- INIT TOKENIZER & HOLDOUT ---
    print("⚙️ Initializing Tokenizer & Holdout Dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.padding_side != 'left' and "xlnet" in model_name:
         tokenizer.padding_side = 'left'

    special_tokens_dict = {'additional_special_tokens': ["<E>", "</E>"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    tok_kwargs = {"max_length": max_length, "truncation": True, "padding": "max_length"} if max_length else {}

    holdout_dataset = PropagandaDataset(df_holdout['context'].tolist(), [label2id[l] for l in df_holdout['label']], tokenizer, **tok_kwargs)
    
    # --- PREPARE HF API ---
    api = HfApi()
    try:
        api.create_repo(repo_id=hf_model_repo, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Repo creation warning: {e}")

    # --- CV LOOP ---
    print(f"✂️ Starting Cross-Validation on {n_folds} Pre-defined Folds...")
    
    for fold in range(n_folds):
        print(f"\n{'='*20} FOLD {fold+1}/{n_folds} {'='*20}")
        
        train_df = df_main[df_main['fold'] != fold].reset_index(drop=True)
        val_df = df_main[df_main['fold'] == fold].reset_index(drop=True)

        print("-" * 40)
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
        print_distribution(train_df, "TRAIN SET")
        print_distribution(val_df, "VAL SET")
        print("-" * 40)

        # --- DATASETS & WEIGHTS  ---
        train_dataset = PropagandaDataset(train_df['context'].tolist(), [label2id[l] for l in train_df['label']], tokenizer, **tok_kwargs)
        val_dataset = PropagandaDataset(val_df['context'].tolist(), [label2id[l] for l in val_df['label']], tokenizer, **tok_kwargs)

        print("⚖️ Calculating Class Weights...")
        train_labels = [x['labels'].item() for x in train_dataset]
        class_weights_arr = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = torch.tensor(class_weights_arr, dtype=torch.float).to(device)

        # --- MODEL & TRAINING  ---
        wandb.init(project="propaganda-detector", name=f"{run_prefix}-fold{fold}", group=run_id, reinit=True)
        wandb.config.update({"model_name": model_name, "fold_index": fold})

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=len(labels_list),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            problem_type="single_label_classification"
        )
        model.resize_token_embeddings(len(tokenizer))

        training_args = TrainingArguments(
            output_dir=f"./results/fold{fold}",
            run_name=f"{run_id}-fold{fold}",
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=batch_size*2,
            eval_strategy="epoch",
            save_strategy="epoch",
            report_to="wandb",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            save_total_limit=1,
            fp16=True,
            push_to_hub=False
        )

        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        print("🚀 Starting training...")
        trainer.train()

        print("Generating OOF predictions...")
        val_preds_output = trainer.predict(val_dataset)
        val_logits = val_preds_output.predictions
        
        fold_oof_df = pd.DataFrame(val_logits, columns=[f"logit_{i}" for i in range(len(labels_list))])
        fold_oof_df['article_id'] = val_df['article_id'].values
        fold_oof_df['start_offset'] = val_df['start_offset'].values
        fold_oof_df['true_label'] = val_df['label'].values
        fold_oof_df['fold'] = fold
        oof_preds_list.append(fold_oof_df)

        print("Generating Holdout predictions...")
        holdout_preds_output = trainer.predict(holdout_dataset)
        holdout_logits_sum += holdout_preds_output.predictions

        print("📊 Logging Confusion Matrix...")
        log_confusion_matrix(trainer, val_dataset, id2label)

        # === SAVING FOLD MODEL ===
        print(f"💾 Saving Fold {fold} model to branch 'fold-{fold}'...")
        fold_save_path = f"./results/fold{fold}_model"
        trainer.save_model(fold_save_path)
        tokenizer.save_pretrained(fold_save_path)

        try:
            api.create_branch(repo_id=hf_model_repo, branch=f"fold-{fold}", exist_ok=True)
        except Exception as e:
            print(f"⚠️ Branch creation warning: {e}")
        
        api.upload_folder(
            folder_path=fold_save_path,
            repo_id=hf_model_repo,
            path_in_repo=".",
            revision=f"fold-{fold}",
            commit_message=f"Training checkpoint: Fold {fold}"
        )
        print(f"✅ Fold {fold} saved to branch 'fold-{fold}'")

        # === SAVING BEST MODEL ===
        current_f1 = val_preds_output.metrics['test_f1_macro']
        print(f"📉 Fold {fold} F1 Macro: {current_f1:.4f}")
        wandb.log({"final_f1_macro": current_f1})

        if current_f1 > best_f1_global:
            print(f"NEW GLOBAL BEST! (F1={current_f1:.4f}) Updating 'main' branch...")
            best_f1_global = current_f1
            api.upload_folder(
                folder_path=fold_save_path,
                repo_id=hf_model_repo,
                path_in_repo=".",
                revision="main",
                commit_message=f"New Best Model: Fold {fold} (F1={current_f1:.4f})"
            )
            
        try:
            shutil.rmtree(fold_save_path)
        except:
            pass
        
        wandb.finish()
        del model, trainer, class_weights
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Cleaning up disk space for Fold {fold}...")
        try:
            shutil.rmtree(training_args.output_dir)
            print(f"✅ Deleted local checkpoint: {training_args.output_dir}")
        except Exception as e:
            print(f"⚠️ Could not cleanup: {e}")

    print("\n💾 Saving Final Datasets & Metadata...")
    safe_model_name = model_name.replace("/", "-")
    
    # 1. Save OOF
    all_oof_df = pd.concat(oof_preds_list, axis=0)
    oof_filename = f"oof_{safe_model_name}.csv"
    all_oof_df.to_csv(oof_filename, index=False)
    
    # 2. Save Holdout
    avg_holdout_logits = holdout_logits_sum / n_folds
    holdout_df = pd.DataFrame(avg_holdout_logits, columns=[f"logit_{i}" for i in range(len(labels_list))])
    holdout_df['article_id'] = df_holdout['article_id'].values
    holdout_df['start_offset'] = df_holdout['start_offset'].values
    holdout_df['true_label'] = df_holdout['label'].values
    holdout_filename = f"holdout_{safe_model_name}.csv"
    holdout_df.to_csv(holdout_filename, index=False)


    print(f"☁️ Uploading artifacts to HF Dataset: {hf_oof_dataset_repo}")
    try:
        for fname in [oof_filename, holdout_filename]:
            api.upload_file(
                path_or_fileobj=fname,
                path_in_repo=fname,
                repo_id=hf_oof_dataset_repo,
                repo_type="dataset",
                token=HF_TOKEN
            )
        print("✅ SUCCESS! All files uploaded to HF.")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

    # --- SANITY CHECK (FINAL ENSEMBLE SCORE) ---
    print("\n Validating Ensemble (Average of 5 folds) on Holdout Set...")
    holdout_preds_indices = np.argmax(avg_holdout_logits, axis=1)
    holdout_true_labels = df_holdout['label'].map(label2id).values
    
    from sklearn.metrics import f1_score, classification_report
    final_ensemble_f1 = f1_score(holdout_true_labels, holdout_preds_indices, average='macro')
    
    print(f"🔥 FINAL HOLDOUT F1 SCORE (Ensemble): {final_ensemble_f1:.4f}")
    print("-" * 30)
    print(classification_report(holdout_true_labels, holdout_preds_indices, target_names=labels_list, digits=4))

    print(f"✅ Pipeline Finished for {model_name}")