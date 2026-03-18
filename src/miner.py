import os
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from huggingface_hub import login
from src.utils.preprocessor import get_tokenize_fn
from datasets import DatasetDict


def mine_hard_examples(
    model_name_or_path="hannusia123123/sc-roberta-miner",
    dataset_name="hannusia123123/propaganda-detector-dataset",
    push_to_hub_name="hannusia123123/propaganda-dataset-hnpm",
    duplication_factor=3,
    max_length=256
):
    print("🚀 Starting Hard Negative/Positive Mining...")
    
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("⚠️ Warning: HF_TOKEN not found in environment.")

    print(f"📥 Loading original training dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, "si_sc_dataset", split="train", token=hf_token)
    
    print(f"📥 Loading miner model: {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    
    tokenize_fn = get_tokenize_fn(tokenizer=tokenizer, max_length=max_length)
        
    print("⚙️ Tokenizing data for inference...")
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    
    training_args = TrainingArguments(
        output_dir="./tmp_miner", 
        per_device_eval_batch_size=32, 
        report_to="none"
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator)
    
    print("🔍 Mining data (running inference on train set)...")
    predictions = trainer.predict(tokenized_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
    probs_class_1 = probs[:, 1]
    y_true = predictions.label_ids
    
    # False Positives
    fp_indices = np.where((y_true == 0) & (probs_class_1 >= 0.5))[0]
    
    # False Negatives
    fn_indices = np.where((y_true == 1) & (probs_class_1 < 0.5))[0]
    
    print(f"⛏️ Found {len(fp_indices)} Hard Negatives (False Positives)")
    print(f"⛏️ Found {len(fn_indices)} Hard Positives (False Negatives)")
    
    hard_negatives = dataset.select(fp_indices)
    hard_positives = dataset.select(fn_indices)
    
    datasets_to_combine = [dataset]
    for _ in range(duplication_factor):
        if len(hard_negatives) > 0:
            datasets_to_combine.append(hard_negatives)
        if len(hard_positives) > 0:
            datasets_to_combine.append(hard_positives)
            
    enhanced_dataset = concatenate_datasets(datasets_to_combine)
    
    print(f"📊 Original dataset size: {len(dataset)}")
    print(f"📊 Enhanced dataset size: {len(enhanced_dataset)}")
    
    print("📋 Fetching original validation split...")
    val_dataset = load_dataset(dataset_name, "si_sc_dataset", split="validation", token=hf_token)
    
    final_dataset = DatasetDict({
        "train": enhanced_dataset,
        "validation": val_dataset
    })
    
    print(f"☁️ Pushing enhanced config to: {dataset_name}...")
    final_dataset.push_to_hub(
        dataset_name, 
        config_name="sc_hnpm_dataset"
    )
    print("✅ Done! Configuration 'sc_hnpm_dataset' is now available in your main repo.")