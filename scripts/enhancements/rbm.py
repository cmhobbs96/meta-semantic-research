import re
import pandas as pd
from typing import List
from pathlib import Path
from scripts.util.savers import save_dataset

def apply_rules_to_dataset(input_path: str, output_path: str):
    """
    Apply rule-based validation to a dataset and remove invalid rows.
    Save the filtered dataset to output_path.
    """
    df = pd.read_csv(input_path, sep="\t", header=None)

    if df.shape[1] == 3:
        df.columns = ["input", "output", "roles"]
    elif df.shape[1] == 2:
        df.columns = ["input", "output"]
    else:
        raise ValueError(f"Unexpected number of columns in {input_path}: {df.shape[1]}")

    # Apply validation rules
    df["valid"] = df["output"].apply(apply_rules_to_prediction)
    filtered_df = df[df["valid"]].drop(columns=["valid"])

    print(f"[RBM] {len(filtered_df)} / {len(df)} rows retained after rule filtering.")

    if filtered_df.empty:
        raise ValueError(f"[RBM] All rows were removed by rule-based validation for: {input_path}")

    # Save result using consistent root path structure
    save_dataset(filtered_df, enhancement="rbm", split=Path(output_path).stem)

def enforce_argument_distinctness(logical_form: str) -> bool:
    """
    Rule: No duplicate arguments in the same predicate (e.g., agent(x, x) is invalid)
    """
    pairs = re.findall(r"\w+\(\s*([^,]+)\s*,\s*([^)]+)\)", logical_form)
    return all(a != b for a, b in pairs)

def require_roles_present(logical_form: str) -> bool:
    """
    Rule: Must contain at least one of the expected roles: agent or theme
    """
    required_roles = {"agent", "theme"}
    found_roles = {match.group(1) for match in re.finditer(r"(\w+)\(", logical_form)}
    return len(required_roles & found_roles) >= 1

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
