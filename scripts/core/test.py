from typing import Tuple
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader
from scripts.core.eval import evaluate
from scripts.util.constants import DEVICE, BATCH_SIZE
from scripts.util.savers import save_predictions, save_metrics
from scripts.util.logger import logger

def test(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    dataset,
    model_name: str,
    enhancement: str,
    dataset_type: str = "test"
) -> Tuple[list, dict]:
    """
    Generate predictions and evaluate a trained T5 model on a given dataset.

    Args:
        model: Trained T5 model
        tokenizer: Tokenizer instance
        dataset: COGSDataset instance (test or gen)
        model_name: Model label (e.g., "t5-small")
        enhancement: Dataset enhancement type ("raw", "srl", etc.)
        dataset_type: "test" or "gen"

    Returns:
        predictions list, metrics dict
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded)

    references = dataset.data["output"].tolist()
    df_metrics, metrics = evaluate(predictions, references)

    save_predictions(predictions, model_name, enhancement, dataset_type=dataset_type)
    save_metrics(df_metrics, model_name, enhancement, dataset_type=dataset_type)

    logger(metrics, title=f"{enhancement} {dataset_type.capitalize()} Eval")

    return predictions, metrics
