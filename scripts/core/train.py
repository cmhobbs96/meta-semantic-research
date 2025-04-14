import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from scripts.util.constants import (
    DEVICE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, ACCUMULATION_STEPS
)
from scripts.util.logger import logger
from scripts.enhancements.srt import compute_role_weighted_loss
from scripts.core.eval import evaluate
from transformers import T5Tokenizer

def train(model, tokenizer, train_dataset, dev_dataset, enhancement: str, role_flag: bool = False):
    """
    Train a T5 model with optional role-aware loss and dev set evaluation.
    """
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler() if torch.cuda.is_available() else None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / ACCUMULATION_STEPS

            if role_flag and "roles" in batch:
                loss = compute_role_weighted_loss(loss, batch["roles"])

            if scaler:
                scaler.scale(loss).backward()
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger(avg_loss, title=f"{enhancement} Train Loss (Epoch {epoch + 1})")

        # --- Validation ---
        model.eval()
        dev_preds, dev_refs = [], []

        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128
                )

                preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                refs = tokenizer.batch_decode(
                    [label[label != -100] for label in labels],
                    skip_special_tokens=True
                )

                dev_preds.extend(preds)
                dev_refs.extend(refs)

        _, dev_metrics = evaluate(dev_preds, dev_refs)
        logger(dev_metrics, title=f"{enhancement} Dev Eval (Epoch {epoch + 1})")

        model.train()

    return model, tokenizer
