import os

def get_dataset_path(enhancement: str, split: str) -> str:
    """
    Get path to data/{enhancement}/{split}.tsv
    """
    return os.path.join("data", enhancement, f"{split}.tsv")

def get_model_path(model_name: str, enhancement: str) -> str:
    """
    Get path to models/{enhancement}/{model_name}/
    """
    return os.path.join("models", enhancement, model_name)

def get_prediction_path(model_name: str, dataset_type: str, enhancement: str) -> str:
    """
    Get path to predictions/{enhancement}/{model_name}_{dataset_type}_predictions.txt
    """
    assert dataset_type in {"test", "gen"}
    return os.path.join("predictions", enhancement, f"{model_name}_{dataset_type}_predictions.txt")

def get_results_path(model_name: str, dataset_type: str, enhancement: str) -> str:
    """
    Get path to results/{enhancement}/{model_name}_{dataset_type}_metrics.csv
    """
    assert dataset_type in {"test", "gen"}
    return os.path.join("results", enhancement, f"{model_name}_{dataset_type}_metrics.csv")

def get_dataset_paths(enhancement: str) -> dict:
    """
    Return dictionary of standard dataset paths
    """
    return {
        "train": get_dataset_path(enhancement, "train"),
        "dev": get_dataset_path(enhancement, "dev"),
        "test": get_dataset_path(enhancement, "test"),
        "gen": get_dataset_path(enhancement, "gen")
    }
