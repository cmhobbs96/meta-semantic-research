import os
import pandas as pd
from pathlib import Path
from .get_root_dir import get_project_root
from transformers import PreTrainedTokenizer, PreTrainedModel

PROJECT_ROOT = get_project_root()

def save_dataset(df: pd.DataFrame, enhancement: str, split: str):
    """
    Save dataset to data/{enhancement}/{split}.tsv
    """
    path = PROJECT_ROOT / "data" / enhancement
    path.mkdir(parents=True, exist_ok=True)

    filepath = path / f"{split}.tsv"
    df.to_csv(filepath, sep="\t", index=False, header=False)
    print(f"Dataset saved to: {filepath}")

def save_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, model_name: str, enhancement: str):
    """
    Save HuggingFace model and tokenizer to models/{enhancement}/{model_name}
    """
    path = PROJECT_ROOT / "models" / enhancement / model_name
    path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model saved to: {path}")

def save_predictions(predictions: list, model_name: str, enhancement: str, dataset_type: str = "test"):
    """
    Save predictions to predictions/{enhancement}/{model_name}_{dataset_type}_predictions.txt
    """
    assert dataset_type in {"test", "gen"}
    path = PROJECT_ROOT / "predictions" / enhancement
    path.mkdir(parents=True, exist_ok=True)

    filepath = path / f"{model_name}_{dataset_type}_predictions.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")

    print(f"Predictions saved to: {filepath}")

def save_metrics(df: pd.DataFrame, model_name: str, enhancement: str, dataset_type: str = "test"):
    """
    Save metrics CSV to results/{enhancement}/{model_name}_{dataset_type}_metrics.csv
    """
    path = PROJECT_ROOT / "results" / enhancement
    path.mkdir(parents=True, exist_ok=True)

    filepath = path / f"{model_name}_{dataset_type}_metrics.csv"
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to: {filepath}")
