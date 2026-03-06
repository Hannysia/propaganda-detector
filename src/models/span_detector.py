from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logits, labels, mask):
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        mask = mask.view(-1)

        active_idx = mask & (labels != -100)
        active_logits = logits[active_idx]
        active_labels = labels[active_idx]

        if len(active_labels) == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)

        ce_loss = F.cross_entropy(active_logits, active_labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()

class PropagandaSpanDetector(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", num_labels=3):
        super(PropagandaSpanDetector, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)        
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.config.hidden_size
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        dropout_prob = getattr(self.config, "hidden_dropout_prob", getattr(self.config, "dropout", 0.1))
        self.dropout = nn.Dropout(dropout_prob)
        
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        self.crf = CRF(num_labels, batch_first=True)
        self.focal_loss = FocalLoss(gamma=2.0)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        lstm_output, _ = self.bilstm(sequence_output)

        lstm_output = self.dropout(lstm_output)
        emissions = self.classifier(lstm_output)

        crf_mask = attention_mask.bool()
        
        if labels is not None:
            clean_labels = labels.clone()
            clean_labels[labels == -100] = 0
                        
            crf_loss = -self.crf(emissions, clean_labels, mask=crf_mask, reduction='mean')
            focal_loss = self.focal_loss(emissions, labels, crf_mask)

            if self.training and torch.rand(1).item() < 0.05:
                print(f"DEBUG: CRF={crf_loss.item():.4f}, Focal={focal_loss.item():.4f}")

            total_loss = crf_loss + focal_loss

            return total_loss
        else:
            preds = self.crf.decode(emissions, mask=crf_mask)
            
            seq_len = emissions.size(1)
            padded_preds = []
            for p in preds:
                padded_preds.append(p + [-100] * (seq_len - len(p)))
                
            return torch.tensor(padded_preds, device=emissions.device)