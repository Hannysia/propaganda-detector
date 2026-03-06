import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import torch
from src.utils.preprocessor import extract_spans_from_tags, merge_close_spans


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


def log_confusion_matrix(trainer, eval_dataset, id2label):
    
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    class_names = [id2label[i] for i in range(len(id2label))]

    cm = confusion_matrix(labels, preds, normalize='true')

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cmap='Blues'
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    wandb.log({"confusion_matrix_img": wandb.Image(plt)})
    plt.close()


def f1_overlap(preds, gold):
    precisions = []
    recalls = []

    for p, g in zip(preds, gold):
        if len(p) == 0 and len(g) == 0:
            precisions.append(1.0); recalls.append(1.0)
            continue
        if len(p) == 0 or len(g) == 0:
            precisions.append(0.0); recalls.append(0.0)
            continue
            
        intersect = len(p.intersection(g))
        precisions.append(intersect / len(p))
        recalls.append(intersect / len(g))

    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def get_exact_match_metrics(pred_spans_list, true_spans_list):
    exact_tp = exact_fp = exact_fn = 0
    for p_spans, t_spans in zip(pred_spans_list, true_spans_list):
        p_set, t_set = set(p_spans), set(t_spans)
        exact_tp += len(p_set.intersection(t_set))
        exact_fp += len(p_set - t_set)
        exact_fn += len(t_set - p_set)
        
    precision = exact_tp / (exact_tp + exact_fp) if (exact_tp + exact_fp) > 0 else 0.0
    recall = exact_tp / (exact_tp + exact_fn) if (exact_tp + exact_fn) > 0 else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def log_si_confusion_matrix(flat_labels, flat_preds):
    
    id2label = {0: "O", 1: "B-PROP", 2: "I-PROP"}
    class_names = [id2label[i] for i in range(3)]

    cm = confusion_matrix(flat_labels, flat_preds, labels=[0, 1, 2], normalize='true')

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='.2f', 
        xticklabels=class_names, yticklabels=class_names, cmap='Blues'
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix (Token Level)')
    plt.tight_layout()

    wandb.log({"confusion_matrix_img": wandb.Image(plt)})
    plt.close()


def compute_si_metrics(eval_preds, eval_dataset, merge_threshold=0):
    predictions, labels = eval_preds.predictions, eval_preds.label_ids
    
    all_pred_indices, all_true_indices = [], []
    all_pred_spans, all_true_spans = [], []
    flat_true_tags, flat_pred_tags = [], []

    for i in range(len(predictions)):
        pred_tags = predictions[i]
        true_tags = labels[i]
        offsets = eval_dataset[i]['offset_mapping'] 
        
        for p_tag, t_tag in zip(pred_tags, true_tags):
            if t_tag != -100:
                flat_pred_tags.append(p_tag)
                flat_true_tags.append(t_tag)
        
        raw_pred_spans = extract_spans_from_tags(pred_tags, offsets)
        
        if merge_threshold > 0:
            pred_spans = merge_close_spans(raw_pred_spans, threshold=merge_threshold)
        else:
            pred_spans = raw_pred_spans
        
        true_spans = extract_spans_from_tags(true_tags, offsets)

        all_pred_spans.append(pred_spans)
        all_true_spans.append(true_spans)
        
        all_pred_indices.append(set(char for start, end in pred_spans for char in range(start, end)))
        all_true_indices.append(set(char for start, end in true_spans for char in range(start, end)))

    sym_precision, sym_recall, sym_f1 = f1_overlap(all_pred_indices, all_true_indices)
    exact_precision, exact_recall, exact_f1 = get_exact_match_metrics(all_pred_spans, all_true_spans)
    
    log_si_confusion_matrix(flat_true_tags, flat_pred_tags)

    return {
        "precision_symbolic": sym_precision,
        "recall_symbolic": sym_recall,
        "f1_symbolic": sym_f1,
        "precision_exact": exact_precision,
        "recall_exact": exact_recall,
        "f1_exact": exact_f1
    }