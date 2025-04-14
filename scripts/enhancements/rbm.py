import re
import pandas as pd
from typing import List
from pathlib import Path
from scripts.util.savers import save_dataset

# Global rule violation counters
rule_counters = {"duplicate_args": 0, "missing_roles": 0}

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
    Apply all rule templates to a single prediction.
    Returns True if all rules pass, False otherwise.
    """
    valid = True
    if not enforce_argument_distinctness(prediction):
        rule_counters["duplicate_args"] += 1
        valid = False
    if not require_roles_present(prediction):
        rule_counters["missing_roles"] += 1
        valid = False
    return valid

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

    df["valid"] = df["roles"].apply(apply_rules_to_prediction)
    filtered_df = df[df["valid"]].drop(columns=["valid"])
    rejected_df = df[~df["valid"]].drop(columns=["valid"])

    print(f"\n[RBM] Loaded: {input_path}")
    print(f"[RBM] Retained {len(filtered_df)} / {len(df)} rows")
    print(f"[RBM] Rejected rows: {len(rejected_df)}")

    if filtered_df.empty:
        raise ValueError(f"[RBM] All rows were removed by rule-based validation for: {input_path}")

    # Show sample rejected rows
    if not rejected_df.empty:
        print("\n[RBM] Sample rejected outputs:")
        print(rejected_df["output"].head(5).to_string(index=False))

        # Save rejected samples for review
        rejected_path = Path(output_path).parent / f"rejected_{Path(output_path).stem}.tsv"
        rejected_df.to_csv(rejected_path, sep="\t", index=False, header=False)
        print(f"[RBM] Rejected samples saved to: {rejected_path}")

    # Show rule violation stats
    print("\n[RBM] Rule violation summary:")
    for rule, count in rule_counters.items():
        print(f"  {rule}: {count}")

    save_dataset(filtered_df, enhancement="rbm", split=Path(output_path).stem)

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
