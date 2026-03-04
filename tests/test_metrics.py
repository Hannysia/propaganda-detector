import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.metrics import f1_overlap, get_exact_match_metrics
from src.utils.preprocessor import extract_spans_from_tags

def test_extract_spans():
    tags = [-100, 0, 1, 2, 0, 2, 0, 1]
    offsets = [(0,0), (0,5), (6,10), (11,15), (16,20), (21,25), (26,30), (31,35)]
    expected_spans = [(6, 15), (21, 25), (31, 35)]
    actual_spans = extract_spans_from_tags(tags, offsets)
    
    assert actual_spans == expected_spans, f"Error: expected {expected_spans}, got {actual_spans}"

def test_f1_overlap():
    # Perfect match
    p, r, f = f1_overlap([set([1,2,3])], [set([1,2,3])])
    assert p == 1.0 and r == 1.0 and f == 1.0

    # Partial match
    p, r, f = f1_overlap([set([1,2,3,4])], [set([3,4,5,6])])
    assert p == 0.5 and r == 0.5 and f == 0.5

    # Empty sets
    p, r, f = f1_overlap([set()], [set()])
    assert p == 1.0 and r == 1.0 and f == 1.0

    # False Positive
    p, r, f = f1_overlap([set([1])], [set()])
    assert p == 0.0 and r == 0.0 and f == 0.0

def test_exact_match():
    # Mix of TP, FP, FN
    preds = [[(0,5), (20,25)]]
    gold = [[(0,5), (10,15)]]
    p, r, f = get_exact_match_metrics(preds, gold)
    assert p == 0.5 and r == 0.5 and f == 0.5

    # Strict boundary mismatch
    preds2 = [[(0,6)]]
    gold2 = [[(0,5)]]
    p2, r2, f2 = get_exact_match_metrics(preds2, gold2)
    assert p2 == 0.0 and r2 == 0.0 and f2 == 0.0

if __name__ == "__main__":
    test_extract_spans()
    test_f1_overlap()
    test_exact_match()
    print("All tests passed.")