# Meta-Semantic Modeling for Compositional Generalization in COGS

This research investigates compositional generalization in deep learning models using the COGS dataset. Our approach enhances the dataset and training process through:

- **Semantic Role Labeling (SRL)**: Agent, Theme, Recipient roles extracted and injected.
- **Knowledge Graph Augmentation (KGA)**: Logical forms transformed into graph-based string representations.
- **Rule-Based Methods (RBM)**: Constraints enforced on logical forms post-prediction.
- **Semantic Role Training (SRT)**: Architecture or loss augmented using role embeddings.
- **Final Pipeline**: Unified enhancement combining SRL + KGA + RBM + SRT.

## Project Structure

meta-semantics-research/
├── data/                 # Raw and enhanced datasets
│   ├── raw/ srl/ kga/ rbm/ srt/ final/
├── models/               # Trained models by enhancement
├── predictions/          # Generated predictions by enhancement
├── results/              # Evaluation metrics by enhancement
├── scripts/
│   ├── utils/            # Logging, constants, paths, memory
│   ├── dataset/          # COGSDataset and path loading
│   ├── metrics/          # Metric functions
│   ├── train.py          # Training pipeline
│   ├── predict.py        # Generation pipeline
│   └── eval.py           # Evaluation metrics
├── notebooks/
│   ├── baseline_metrics/
│   ├── srl_annotation/
│   ├── knowledge_graph_augmentation/
│   ├── semantic_role_training/
│   ├── rule_based_methods/
│   └── final/
└── README.md

## Tools
- Model: T5
- Frameworks: PyTorch, Hugging Face Transformers
- Platform: Paperspace Notebooks + GitHub Integration
- Evaluation: BLEU, Exact Match, Role F1, Predicate Match, Hallucination Rate

## Goals
1. Build reproducible training + evaluation scripts
2. Run modular experiments for each enhancement
3. Compare results and visualize trends
4. Publish final results as a research artifact

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/meta-semantic-cogs.git
cd meta-semantic-cogs
pip install -r requirements.txt