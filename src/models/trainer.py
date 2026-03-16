import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, loss_type="cross_entropy", gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        logits = logits.to(torch.float32)
        weight = self.class_weights.to(device=model.device, dtype=torch.float32)
        
        if self.loss_type == "focal":

            ce_loss_unweighted = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1).long(), 
                reduction='none'
            )
            pt = torch.exp(-ce_loss_unweighted)
            
            ce_loss_weighted = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1).long(), 
                weight=weight, 
                reduction='none'
            )
            
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss_weighted
            loss = focal_loss.mean()
            
        else:
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1).long()
            )
        
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