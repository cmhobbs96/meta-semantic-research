# scripts/metrics/visuals.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.util.getters import get_results_path

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

    # Drop columns that shouldn't be plotted
    drop_cols = ["elapsed_time"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    # Melt for bar plotting
    df_long = df.melt(var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_long, x="Metric", y="Score", palette="crest")
    plt.title(f"{model_name} — Evaluation Metrics")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

def compare_test_gen_metrics(model_name: str, enhancement: str = "raw"):
    """
    Compare test vs gen metrics for a given model and enhancement.
    
    Args:
        model_name (str): Name of the model (e.g. "t5-small")
        enhancement (str): Enhancement type (e.g. "raw", "srl", etc.)
    """
    # Use your centralized path resolver
    test_path = get_results_path(model_name, "test", enhancement)
    gen_path  = get_results_path(model_name, "gen", enhancement)

    # Load data
    test_df = pd.read_csv(test_path)
    gen_df  = pd.read_csv(gen_path)

    # Add split labels
    test_df["Split"] = "Test"
    gen_df["Split"] = "Gen"

    # Combine and reshape
    df_all = pd.concat([test_df, gen_df], ignore_index=True)

    # Get all metric columns (exclude 'Split' if it's present)
    metric_columns = [col for col in df_all.columns if col != "Split"]

    # Plot one figure per metric
    for metric in metric_columns:
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df_all, x="Split", y=metric, palette="muted")
        plt.title(f"{model_name.upper()} — {enhancement.upper()} | {metric.replace('_', ' ').title()} Comparison")
        plt.ylabel("Score")
        plt.ylim(0, max(df_all[metric]) * 1.2 if df_all[metric].max() > 0 else 1)
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()