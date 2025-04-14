import pandas as pd
import torch
from torch.utils.data import Dataset
from scripts.util.constants import MAX_LENGTH

class COGSDataset(Dataset):
    def __init__(self, file_path, tokenizer, sample_frac=None, inject_roles=True, role_flag=False):
        """
        Args:
            file_path (str): Path to the .tsv dataset file
            tokenizer: Hugging Face tokenizer
            sample_frac (float): Sample fraction (for debug mode)
            inject_roles (bool): Whether to include roles in the input string
            role_flag (bool): Whether to return raw role string (for semantic role training)
        """
        df = pd.read_csv(file_path, sep="\t", header=None)

        if inject_roles:
            if df.shape[1] >= 3:
                df = df.iloc[:, :3]
                df.columns = ["input", "output", "roles"]
            else:
                raise ValueError(f"Expected 3 columns for SRL mode, got {df.shape[1]}")
        else:
            df = df.iloc[:, :2]
            df.columns = ["input", "output"]

        if sample_frac:
            df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

        self.data = df
        self.tokenizer = tokenizer
        self.inject_roles = inject_roles
        self.role_flag = role_flag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Inject SRL into input string
        input_text = f"{row['input']} [ROLES] {row['roles']}" if (
            self.inject_roles and "roles" in row and pd.notna(row["roles"])
        ) else row["input"]

        output_text = row["output"]

        # Tokenize input/output
        inputs = self.tokenizer(
            input_text, padding="max_length", truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        )
        targets = self.tokenizer(
            output_text, padding="max_length", truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        )

        labels = targets["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        item = {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels
        }

        if self.role_flag and "roles" in row:
            item["roles"] = row["roles"]  # return raw role string

        return item
