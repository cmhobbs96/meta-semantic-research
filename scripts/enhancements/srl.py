import re
import pandas as pd
from scripts.util.savers import save_dataset

def extract_roles_from_logical_form(logical_form: str) -> str:
    """
    Extract agent, theme, recipient roles from a logical form and return them as a semicolon-separated string.
    """
    roles = []
    role_types = ["agent", "theme", "recipient", "location"]

    for role in role_types:
        matches = re.findall(fr"{role}\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", logical_form)
        for a, b in matches:
            roles.append(f"{role}({a},{b})")

    return "; ".join(roles) if roles else "none"

def annotate_dataset_with_srl(input_path: str, output_path: str):
    """
    Annotate a raw dataset with extracted SRL roles and save the result.
    """
    df = pd.read_csv(input_path, sep="\t", header=None)
    if df.shape[1] == 3:
        df.columns = ["input", "output", "split"]
    elif df.shape[1] == 2:
        df.columns = ["input", "output"]
    else:
        raise ValueError(f"Unexpected number of columns in {input_path}: {df.shape[1]}")

    df["roles"] = df["output"].apply(extract_roles_from_logical_form)
    df = df[["input", "output", "roles"]]
    save_dataset(df, enhancement="srl", split=Path(output_path).stem)
