import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, fbeta_score
import torch
from src.utils.preprocessor import extract_spans_from_tags, merge_close_spans


def compute_metrics(eval_pred, average='macro', pos_label=1, include_fbeta=False):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    kwargs = {"average": average, "zero_division": 0}
    if average == 'binary':
        kwargs["pos_label"] = pos_label
    
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, **kwargs),
        "recall": recall_score(labels, preds, **kwargs),
        "f1": f1_score(labels, preds, **kwargs),
    }
    
    if include_fbeta:
        metrics["f2"] = fbeta_score(labels, preds, beta=2.0, **kwargs)
        metrics["f3"] = fbeta_score(labels, preds, beta=3.0, **kwargs)
        
    return metrics


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


def f1_overlap(all_preds, all_gold):

    S_total = [s for s in all_preds if len(s) > 0]
    T_total = [t for t in all_gold if len(t) > 0]

    if not S_total and not T_total:
        return 1.0, 1.0, 1.0
    if not S_total or not T_total:
        return 0.0, 0.0, 0.0

    p_sum = 0
    for s in S_total:
        s_overlap_sum = 0
        for t in T_total:
            intersect_len = len(s.intersection(t))
            if intersect_len > 0:
                s_overlap_sum += (intersect_len / len(t))
        p_sum += s_overlap_sum
    precision = p_sum / len(S_total)

    r_sum = 0
    for t in T_total:
        t_overlap_sum = 0
        for s in S_total:
            intersect_len = len(t.intersection(s))
            if intersect_len > 0:
                t_overlap_sum += (intersect_len / len(s))
        r_sum += t_overlap_sum
    recall = r_sum / len(T_total)

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
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
    all_pred_spans_normalized, all_true_spans_normalized = [], []
    flat_true_tags, flat_pred_tags = [], []

    for i in range(len(predictions)):
        pred_tags = predictions[i]
        true_tags = labels[i]
        offsets = eval_dataset[i]['offset_mapping']
        current_offset = eval_dataset[i]['offset']
        
        for p_tag, t_tag in zip(pred_tags, true_tags):
            if t_tag != -100:
                flat_pred_tags.append(p_tag)
                flat_true_tags.append(t_tag)
        
        valid_indices = [idx for idx, t_tag in enumerate(true_tags) if t_tag != -100]

        if not valid_indices:
            all_pred_indices.append(set())
            all_true_indices.append(set())
            all_pred_spans_normalized.append([])
            all_true_spans_normalized.append([])
            continue

        start_valid_char = offsets[valid_indices[0]][0]
        end_valid_char = offsets[valid_indices[-1]][1]

        raw_pred_spans = extract_spans_from_tags(pred_tags, offsets)
        raw_true_spans = extract_spans_from_tags(true_tags, offsets)

        norm_pred_spans = []
        for s, e in raw_pred_spans:
            s_crop, e_crop = max(s, start_valid_char), min(e, end_valid_char)
            if e_crop > s_crop:
                norm_pred_spans.append((s_crop - current_offset, e_crop - current_offset))

        if merge_threshold > 0:
            norm_pred_spans = merge_close_spans(norm_pred_spans, threshold=merge_threshold)

        norm_true_spans = [(s - current_offset, e - current_offset) for s, e in raw_true_spans]

        p_set = set()
        for s, e in norm_pred_spans: p_set.update(range(s, e))
        
        g_set = set()
        for s, e in norm_true_spans: g_set.update(range(s, e))

        all_pred_indices.append(p_set)
        all_true_indices.append(g_set)
        
        all_pred_spans_normalized.append(norm_pred_spans)
        all_true_spans_normalized.append(norm_true_spans)

    sym_p, sym_r, sym_f1 = f1_overlap(all_pred_indices, all_true_indices)
    exact_p, exact_r, exact_f1 = get_exact_match_metrics(all_pred_spans_normalized, all_true_spans_normalized)
    
    log_si_confusion_matrix(flat_true_tags, flat_pred_tags)

    return {
        "precision_symbolic": sym_p,
        "recall_symbolic": sym_r,
        "f1_symbolic": sym_f1,
        "precision_exact": exact_p,
        "recall_exact": exact_r,
        "f1_exact": exact_f1
    }