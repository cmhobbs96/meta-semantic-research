import re
import pandas as pd
from typing import List

def enforce_argument_distinctness(logical_form: str) -> bool:
    """
    Rule: No duplicate arguments in the same predicate (e.g., agent(x, x) is invalid)
    """
    pairs = re.findall(r"\w+\(\s*([^,]+)\s*,\s*([^)]+)\)", logical_form)
    return all(a != b for a, b in pairs)

def require_roles_present(logical_form: str) -> bool:
    """
    Rule: Must contain at least agent and theme
    """
    roles = {"agent", "theme"}
    found = {match.group(1) for match in re.finditer(r"(\w+)\(", logical_form)}
    return roles.issubset(found)

def apply_rules_to_prediction(prediction: str) -> bool:
    """
    Apply all rule templates to a single prediction
    Returns True if all rules pass, False otherwise
    """
    return (
        enforce_argument_distinctness(prediction) and
        require_roles_present(prediction)
    )

def validate_prediction_rules(predictions: List[str], references: List[str] = None) -> pd.DataFrame:
    """
    Apply rule-based validation to a list of predictions.
    Optionally include references for comparison.
    """
    result = []
    for i, pred in enumerate(predictions):
        valid = apply_rules_to_prediction(pred)
        entry = {"prediction": pred, "rule_valid": valid}
        if references:
            entry["reference"] = references[i]
        result.append(entry)

    return pd.DataFrame(result)
