# Meta-Semantic Research

## Overview
This project investigates compositional generalization in deep learning models by integrating semantic role labeling (SRL) and meta-semantic modeling into the COGS (Compositional Generalization Challenge) dataset. The goal is to improve systematic generalization beyond memorized patterns by:

1. Enhancing the COGS dataset with explicit SRL annotations (Agent, Theme, Recipient).

2. Developing meta-semantic modeling techniques, including:
  - Knowledge Graph Augmentation: Representing sentence structures as knowledge graphs.

  - Hybrid Deep Learning + Rule-Based Methods: Reinforcing linguistic constraints in neural models.

  - Semantic Role-Aware Training: Encoding SRL explicitly in model embeddings and architecture.

## Research Goals
1. **Baseline:** Train and evaluate models on the original and SRL-enhanced COGS dataset.

2. **Meta-Semantic Modeling:** Implement structured knowledge integration.

3. **Evaluation:** Compare results across different modeling approaches to assess improvements in generalization.
    - Exact Match Accuracy
    - F1 Score
    - Compostitional Generalization Score
    - Error Analysis

## Repository Structure
```
meta-semantic-research/
│── data/                     # Raw and processed datasets
│   ├── COGS/
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   ├── train.tsv
│   │   ├── gen.tsv
│   ├── PropBank/
│   ├── UniversalDependencies/
│── models/                   # Model checkpoints
│   ├── baseline_t5.pth
│   ├── meta_kg_t5.pth
│── scripts/                  # Python scripts for processing & training
│   ├── preprocess_srl.py
│   ├── train_baseline.py
│   ├── train_knowledge_graph.py
│── results/                  # Model evaluation results
│   ├── baseline_metrics.json
│   ├── meta_kg_metrics.json
│── notebooks/                # Google Colab notebooks
│   ├── Baseline_COGS.ipynb
│   ├── Baseline_COGS_SRL.ipynb
│   ├── Meta_Semantic_KG.ipynb
│── README.md                 # Project description
│── requirements.txt          # Dependencies (HuggingFace, PyTorch, etc.)
│── .gitignore                # Ignore unnecessary files (e.g., checkpoints)

```

## Datasets
1. **COGS Dataset:** Enhancing with SRL & Knowledge Graphs
2. **PropBank:** Enhancing for Compositional Generalization
3. **Universal Dependencies (UD):** Enhancing with SRL & Logical Structures

## Large Language Models
1. **T5 (Text-to-Text Transfer Transformer):** Assess compositional generalization
2. **BART (Denoising Autoencoder Transformer):** Evaluate on enhanced SRL input
3. **OpenAI GPT-4o-mini:** Compare against explicit SRL and KG-enhanced training
4. **(Optional) LSTMs & Attention:** Traditional baseline evaluation

## Literature Review

## Research Paper Progress

## Future Work

