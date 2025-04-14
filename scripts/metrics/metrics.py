import re
import numpy as np
import pandas as pd
from typing import List, Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Levenshtein import distance as levenshtein_distance

def extract_roles(logical_form: str) -> set:
    return set(re.findall(r"(\w+)\s*\([^,]+,\s*[^)]+\)", logical_form))

def extract_predicates(logical_form: str) -> set:
    return set(re.findall(r"(\w+)\s*\(", logical_form))

def is_valid_logical_form(text: str) -> bool:
    return bool(re.match(r"^[\*\w\s\.\(\),;AND-]+$", text))

def compute_precision(pred_roles: set, ref_roles: set) -> float:
    if not pred_roles:
        return 0.0
    return len(pred_roles & ref_roles) / len(pred_roles)

def compute_recall(pred_roles: set, ref_roles: set) -> float:
    if not ref_roles:
        return 0.0
    return len(pred_roles & ref_roles) / len(ref_roles)

def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def compute_bleu(pred: str, ref: str) -> float:
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)

def compute_predicate_accuracy(pred: str, ref: str) -> float:
    pred_predicates = extract_predicates(pred)
    ref_predicates = extract_predicates(ref)
    return len(pred_predicates & ref_predicates) / len(ref_predicates) if ref_predicates else 0.0
