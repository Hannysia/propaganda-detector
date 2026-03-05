from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
from torchcrf import CRF

class PropagandaSpanDetector(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", num_labels=3):
        super(PropagandaSpanDetector, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        crf_mask = attention_mask.bool()
        
        if labels is not None:
            clean_labels = labels.clone()
            clean_labels[labels == -100] = 0
            
            loss = -self.crf(emissions, clean_labels, mask=crf_mask, reduction='mean')
            return loss
        else:
            preds = self.crf.decode(emissions, mask=crf_mask)
            
            seq_len = emissions.size(1)
            padded_preds = []
            for p in preds:
                padded_preds.append(p + [-100] * (seq_len - len(p)))
                
            return torch.tensor(padded_preds, device=emissions.device)