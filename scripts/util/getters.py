import os
from pathlib import Path
from .get_root_dir import get_project_root

PROJECT_ROOT = get_project_root()

def get_dataset_path(enhancement: str, split: str) -> str:
    """
    Get absolute path to data/{enhancement}/{split}.tsv
    """
    return str(PROJECT_ROOT / "data" / enhancement / f"{split}.tsv")

def get_model_path(model_name: str, enhancement: str) -> str:
    """
    Get absolute path to models/{enhancement}/{model_name}/
    """
    return str(PROJECT_ROOT / "models" / enhancement / model_name)

def get_prediction_path(model_name: str, dataset_type: str, enhancement: str) -> str:
    """
    Get absolute path to predictions/{enhancement}/{model_name}_{dataset_type}_predictions.txt
    """
    assert dataset_type in {"test", "gen"}
    filename = f"{model_name}_{dataset_type}_predictions.txt"
    return str(PROJECT_ROOT / "predictions" / enhancement / filename)

def get_results_path(model_name: str, dataset_type: str, enhancement: str) -> str:
    """
    Get absolute path to results/{enhancement}/{model_name}_{dataset_type}_metrics.csv
    """
    assert dataset_type in {"test", "gen"}
    filename = f"{model_name}_{dataset_type}_metrics.csv"
    return str(PROJECT_ROOT / "results" / enhancement / filename)

def get_dataset_paths(enhancement: str) -> dict:
    """
    Return dict of dataset paths for {train, dev, test, gen}
    """
    return {
        "train": get_dataset_path(enhancement, "train"),
        "dev": get_dataset_path(enhancement, "dev"),
        "test": get_dataset_path(enhancement, "test"),
        "gen": get_dataset_path(enhancement, "gen"),
    }
