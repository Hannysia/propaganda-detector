import torch
from torch.utils.data import Dataset

class PropagandaSIDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.all_windows = []
                
        for item in data:
            prev_t = item['prev_text']
            curr_t = item['text']
            next_t = item['next_text']
            
            article_id = item['article_id']
            original_spans = item['propaganda_spans']
            
            window_text = prev_t + curr_t + next_t
            
            len_prev = len(prev_t)
            len_curr = len(curr_t)
            len_target_end = len_prev + len_curr
            
            window_spans = [{'start': s[0] + len_prev, 'end': s[1] + len_prev} for s in original_spans]
            
            encodings = tokenizer(
                window_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt"
            )
            
            input_ids = encodings["input_ids"][0]
            attention_mask = encodings["attention_mask"][0]
            offset_mapping = encodings["offset_mapping"][0]
            
            labels = torch.full_like(input_ids, -100, dtype=torch.long)
            previous_span_idx = -1 
            
            for j, (start_char, end_char) in enumerate(offset_mapping):

                if start_char == 0 and end_char == 0:
                    previous_span_idx = -1
                    continue
                    
                if end_char <= len_prev or start_char >= len_target_end:
                    previous_span_idx = -1
                    continue
                    
                current_span_idx = -1
                
                for span_idx, span in enumerate(window_spans):
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
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "offset_mapping": offset_mapping,
                "article_id": article_id,
                "offset": len_prev
            })
                
    def __len__(self):
        return len(self.all_windows)

    def __getitem__(self, idx):
        return {
            "input_ids": self.all_windows[idx]["input_ids"],
            "attention_mask": self.all_windows[idx]["attention_mask"],
            "labels": self.all_windows[idx]["labels"],
            "offset_mapping": self.all_windows[idx]["offset_mapping"],
            "article_id": self.all_windows[idx]["article_id"],
            "offset": self.all_windows[idx]["offset"]
        }