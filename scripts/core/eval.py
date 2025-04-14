import pandas as pd
import numpy as np
from typing import List, Tuple
from scripts.metrics.metrics import (
    extract_roles,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_bleu,
    levenshtein_distance,
    compute_predicate_accuracy,
    is_valid_logical_form,
    compute_exact_match
)

def evaluate(preds: List[str], refs: List[str]) -> Tuple[pd.DataFrame, dict]:
    assert len(preds) == len(refs)

    precisions = []
    recalls = []
    f1s = []
    bleus = []
    edits = []
    pred_accs = []
    valid_flags = []
    exact_matches = []

    for pred, ref in zip(preds, refs):
        pred, ref = pred.strip(), ref.strip()
        pred_roles = extract_roles(pred)
        ref_roles = extract_roles(ref)

        p = compute_precision(pred_roles, ref_roles)
        r = compute_recall(pred_roles, ref_roles)
        f1 = compute_f1(p, r)
        bleu = compute_bleu(pred, ref)
        edit = levenshtein_distance(pred, ref)
        pred_acc = compute_predicate_accuracy(pred, ref)
        valid = is_valid_logical_form(pred)
        exact = compute_exact_match(pred, ref)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        bleus.append(bleu)
        edits.append(edit)
        pred_accs.append(pred_acc)
        valid_flags.append(valid)
        exact_matches.append(exact)

    metrics = {
        "exact_match": np.mean(exact_matches),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": np.mean(f1s),
        "bleu": np.mean(bleus),
        "edit_distance_avg": np.mean(edits),
        "predicate_accuracy": np.mean(pred_accs),
        "valid_logical_form_ratio": np.mean(valid_flags),
    }

    return pd.DataFrame([metrics]), metrics
