import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


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


def log_confusion_matrix(trainer, val_dataset, id2label):

    predictions_output = trainer.predict(val_dataset)
    preds_id = np.argmax(predictions_output.predictions, axis=-1)
    labels_id = predictions_output.label_ids
    
    preds_str = [id2label[i] for i in preds_id]
    labels_str = [id2label[i] for i in labels_id]
    class_names = list(id2label.values())

    wandb.log({
        "conf_mat": wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels_id,
            preds=preds_id,
            class_names=class_names
        )
    })
