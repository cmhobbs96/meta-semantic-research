# scripts/metrics/visuals.py

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
    test_path = f"results/{enhancement}/{model_name}_test_metrics.csv"
    gen_path  = f"results/{enhancement}/{model_name}_gen_metrics.csv"

    df_test = pd.read_csv(test_path).T
    df_gen  = pd.read_csv(gen_path).T

    df_test.columns = ['Test']
    df_gen.columns = ['Gen']

    combined = pd.concat([df_test, df_gen], axis=1)

    # Styled Table
    styled = combined.style.format("{:.4f}").set_caption("Metric Comparison — Test vs Gen")
    display(styled)

    # Bar chart per metric
    for metric in combined.index:
        plt.figure(figsize=(6, 4))
        sns.barplot(x=combined.columns, y=combined.loc[metric], palette="deep")
        plt.title(f"{model_name.upper()} — {enhancement.upper()} | {metric}")
        plt.ylabel("Score")
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()