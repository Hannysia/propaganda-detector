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
            
            num_article_windows = encodings["input_tokens_ids"].shape[0]
            
            for i in range(num_article_windows):
                input_tokens_ids = encodings["input_tokens_ids"][i]
                attention_mask = encodings["attention_mask"][i]
                offset_mapping = encodings["offset_mapping"][i]
                
                labels = torch.zeros_like(input_tokens_ids, dtype=torch.long)
                
                previous_span_idx = -1 
                
                for j, (start_char, end_char) in enumerate(offset_mapping):
                    
                    if start_char == 0 and end_char == 0:
                        labels[j] = -100
                        previous_span_idx = -1
                        continue
                        
                    current_span_idx = -1
                    
                    for span_idx, span in enumerate(spans):
                        if start_char < span['end'] and end_char > span['start']: 
                            current_span_idx = span_idx
                            break
                            
                    if current_span_idx != -1:
                        if current_span_idx != previous_span_idx:
                            labels[j] = 1 
                        else:
                            labels[j] = 2 
                    else:
                        labels[j] = 0 
                        
                    previous_span_idx = current_span_idx
                        
                self.all_windows.append({
                    "input_tokens_ids": input_tokens_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "offset_mapping": offset_mapping,
                    "article_id": article_id
                })
                
    def __len__(self):
        return len(self.all_windows)

    def __getitem__(self, idx):
        return {
            "input_tokens_ids": self.all_windows[idx]["input_tokens_ids"],
            "attention_mask": self.all_windows[idx]["attention_mask"],
            "labels": self.all_windows[idx]["labels"],
            "offset_mapping": self.all_windows[idx]["offset_mapping"],
            "article_id": self.all_windows[idx]["article_id"]
        }