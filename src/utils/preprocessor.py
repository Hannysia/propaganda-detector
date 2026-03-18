import re
import spacy
import os

try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    NLP = spacy.load("en_core_web_sm")

def clean_punctuation(text):

    if not isinstance(text, str): return str(text)
    
    text = text.replace('“', '"').replace('”', '"').replace("’", "'").replace("‘", "'")
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace('—', '-')
    
    return text

def normalize_text(text):
    doc = NLP(text)
    text = " ".join([token.text for token in doc])
    
    text = re.sub(r'<\s*E\s*>', '<E>', text)
    text = re.sub(r'<\s*/\s*E\s*>', '</E>', text)
    
    text = re.sub(r'\s+([?.!,:;])', r'\1', text)

    return text.strip()

def get_tagged_context(text, fragment=None, start_char=None, end_char=None):
    if start_char is not None and end_char is not None:
        start, end = start_char, end_char
    elif fragment is not None:
        try:
            start = text.index(fragment)
            end = start + len(fragment)
        except ValueError:
            return None, None, "Fragment not found in the text."
    else:
        return None, None, "Provide either offsets or fragment string."

    raw_fragment = text[start:end]
    final_fragment = clean_punctuation(raw_fragment.replace('\n', ' ').strip())

    doc = NLP(text)
    sentences = list(doc.sents)
    
    start_sent_idx, end_sent_idx = -1, -1
    for i, sent in enumerate(sentences):
        if sent.start_char <= start < sent.end_char:
            start_sent_idx = i
        if sent.start_char < end <= sent.end_char:
            end_sent_idx = i
            
    if start_sent_idx == -1: 
        return None, None, "Error aligning fragment."
    
    if end_sent_idx == -1: end_sent_idx = start_sent_idx

    target_sentences = sentences[start_sent_idx : end_sent_idx + 1]
    span_start = target_sentences[0].start_char
    span_end = target_sentences[-1].end_char
    
    raw_window = text[span_start : span_end]
    rel_start, rel_end = start - span_start, end - span_start
    
    context_tagged = (
        raw_window[:rel_start] + " <E> " + 
        raw_window[rel_start:rel_end] + " </E> " + 
        raw_window[rel_end:]
    )
    
    final_context = normalize_text(clean_punctuation(context_tagged.replace('\n', ' ')))
    
    return final_context, final_fragment, None

def extract_spans_from_tags(tags, offsets):
    """
    Converts BIO tags (0, 1, 2) back to character coordinates (start, end).
    """
    spans = []
    current_start = None
    prev_end = None
    
    for tag, offset in zip(tags, offsets):
        if tag == -100 or offset == [0, 0] or offset == (0, 0):
            continue
            
        start_char, end_char = offset
        
        if tag == 1:
            if current_start is not None:
                spans.append((current_start, prev_end))
            current_start = start_char
        elif tag == 2:
            if current_start is None: 
                current_start = start_char 
        elif tag == 0:
            if current_start is not None:
                spans.append((current_start, prev_end))
                current_start = None
                
        prev_end = end_char
        
    if current_start is not None:
        spans.append((current_start, prev_end))
        
    return spans

def merge_close_spans(spans, threshold):
    """
    Sticks together spans if the distance between them is less than or equal to threshold (in characters).
    """
    if not spans:
        return []
        
    spans = sorted(spans, key=lambda x: x[0])
    merged = [spans[0]]
    
    for current in spans[1:]:
        previous = merged[-1]
        
        if current[0] - previous[1] <= threshold:
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            merged.append(current)
            
    return merged

def get_tokenize_fn(tokenizer, max_length):

    def tokenize_fn(examples):
        combined_texts = []
        for p, t, n in zip(examples["prev_text"], examples["text"], examples["next_text"]):
            p_safe = p if p is not None else ""
            t_safe = t if t is not None else ""
            n_safe = n if n is not None else ""
            
            combined = f"{p_safe} {tokenizer.sep_token} {t_safe} {tokenizer.sep_token} {n_safe}".strip()
            combined = " ".join(combined.split()) 
            combined_texts.append(combined)
            
        return tokenizer(combined_texts, truncation=True, max_length=max_length)
    
    return tokenize_fn