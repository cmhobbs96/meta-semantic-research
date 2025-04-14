import os
import pandas as pd

def save_dataset(df: pd.DataFrame, enhancement: str, split: str):
    """
    Save dataset TSV to data/{enhancement}/{split}.tsv
    """
    path = os.path.join("data", enhancement)
    os.makedirs(path, exist_ok=True)
    output_file = os.path.join(path, f"{split}.tsv")
    df.to_csv(output_file, sep="\t", index=False, header=False)
    print(f"[INFO] Dataset saved to: {output_file}")

def save_model(model, tokenizer, model_name: str, enhancement: str):
    path = os.path.join("models", enhancement, model_name)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"[INFO] Model and tokenizer saved to: {path}")

def save_predictions(predictions: list, model_name: str, enhancement: str, dataset_type="test"):
    assert dataset_type in {"test", "gen"}
    path = os.path.join("predictions", enhancement)
    os.makedirs(path, exist_ok=True)
    output_file = os.path.join(path, f"{model_name}_{dataset_type}_predictions.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")
    print(f"[INFO] Predictions saved to: {output_file}")

def save_metrics(metrics_df: pd.DataFrame, model_name: str, enhancement: str, dataset_type="test"):
    assert dataset_type in {"test", "gen"}
    path = os.path.join("results", enhancement)
    os.makedirs(path, exist_ok=True)
    output_file = os.path.join(path, f"{model_name}_{dataset_type}_metrics.csv")
    metrics_df.to_csv(output_file, index=False)
    print(f"[INFO] Metrics saved to: {output_file}")