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
    Converts BIOES tags (0, 1, 2, 3, 4) back to character coordinates (start, end).
    """
    spans = []
    start_char = None
    last_end_char = None

    for tag, offset in zip(tags, offsets):

        if offset[0] == 0 and offset[1] == 0:
            continue
            
        if tag == 4:
            if start_char is not None:
                spans.append((start_char, last_end_char))
            spans.append((offset[0], offset[1]))
            start_char = None
            
        elif tag == 1:
            if start_char is not None:
                spans.append((start_char, last_end_char))
            start_char = offset[0]
            
        elif tag == 3:
            if start_char is not None:
                spans.append((start_char, offset[1]))
                start_char = None
            else:
                spans.append((offset[0], offset[1]))
                
        elif tag == 0:
            if start_char is not None:
                spans.append((start_char, last_end_char))
                start_char = None
                
        last_end_char = offset[1]

    if start_char is not None and last_end_char is not None:
        spans.append((start_char, last_end_char))
        
    return spans