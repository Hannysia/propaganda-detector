import torch
from torch import nn
from transformers import Trainer

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    

class SITrainer(Trainer):
    def __init__(self, pos_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        
        loss = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            labels=labels
        )

        if labels is not None and (labels > 0).any():
            loss = loss * self.pos_weight

        if return_outputs:
            with torch.no_grad():
                predictions = model(
                    input_ids=inputs.get("input_ids"), 
                    attention_mask=inputs.get("attention_mask")
                )
            return (loss, {"logits": predictions}) 
            
        return loss