import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics_table(csv_path: str):
    """
    Display a styled table of evaluation metrics.
    """
    df = pd.read_csv(csv_path)
    styled = df.style.format("{:.4f}").set_caption("Evaluation Metrics")
    display(styled)

def plot_metrics_bar(csv_path: str, model_name: str = "Model"):
    """
    Plot bar chart of metrics from a saved CSV file.
    """
    df = pd.read_csv(csv_path)

    # Optional: Drop irrelevant columns
    drop_cols = ["elapsed_time"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    df_long = df.melt(var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_long, x="Metric", y="Score", palette="crest")
    plt.title(f"{model_name} — Evaluation Metrics")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()
