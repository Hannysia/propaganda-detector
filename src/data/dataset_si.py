import torch
from torch.utils.data import Dataset

class PropagandaSIDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, stride=200):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.all_windows = []
                
        for item in data:
            text = item['text']
            spans = item['spans']
            article_id = item['article_id']
            
            encodings = tokenizer(
                text,
                max_length=self.max_length,
                stride=self.stride,
                truncation=True,
                padding="max_length",
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_tensors="pt"
            )
            
            num_article_windows = encodings["input_ids"].shape[0]
            
            for i in range(num_article_windows):
                input_ids = encodings["input_ids"][i]
                attention_mask = encodings["attention_mask"][i]
                offset_mapping = encodings["offset_mapping"][i]
                
                labels = torch.zeros_like(input_ids, dtype=torch.long)
                
                token_span_indices = []
                for start_char, end_char in offset_mapping:

                    if start_char == 0 and end_char == 0:
                        token_span_indices.append(-100)
                        continue
                        
                    matched_span = -1
                    for span_idx, span in enumerate(spans):
                        if start_char < span['end'] and end_char > span['start']: 
                            matched_span = span_idx
                            break
                    token_span_indices.append(matched_span)
                    
                for j in range(len(token_span_indices)):
                    span_idx = token_span_indices[j]
                    
                    if span_idx == -100:
                        labels[j] = -100
                        continue
                    if span_idx == -1:
                        labels[j] = 0
                        continue
                        
                    is_start = (j == 0) or (token_span_indices[j-1] != span_idx)
                    is_end = (j == len(token_span_indices) - 1) or (token_span_indices[j+1] != span_idx)
                    
                    if is_start and is_end:
                        labels[j] = 4
                    elif is_start:
                        labels[j] = 1
                    elif is_end:
                        labels[j] = 3
                    else:
                        labels[j] = 2
                        
                self.all_windows.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "offset_mapping": offset_mapping,
                    "article_id": article_id
                })
                
    def __len__(self):
        return len(self.all_windows)

    def __getitem__(self, idx):
        return {
            "input_ids": self.all_windows[idx]["input_ids"],
            "attention_mask": self.all_windows[idx]["attention_mask"],
            "labels": self.all_windows[idx]["labels"],
            "offset_mapping": self.all_windows[idx]["offset_mapping"],
            "article_id": self.all_windows[idx]["article_id"]
        }