# Bridging Deep Learning and Linguistic Structure: A Meta-Semantic Approach to Compositional Generalization
This research investigates compositional generalization in deep learning models using the COGS dataset. Our approach enhances the dataset and training process through:

- **Semantic Role Labeling (SRL)**: Agent, Theme, Recipient roles extracted and injected.
- **Knowledge Graph Augmentation (KGA)**: Logical forms transformed into graph-based string representations.
- **Rule-Based Methods (RBM)**: Constraints enforced on logical forms post-prediction.
- **Semantic Role Training (SRT)**: Architecture or loss augmented using role embeddings.
- **Final Pipeline**: Unified enhancement combining SRL + KGA + RBM + SRT.

## Project Structure
```
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
```
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

## Literature Review
```
Findlay, J. Y., Salimifar, S., Yıldırım, A., and Haug, T. T.
Rule-based semantic interpretation for universal depen-
dencies. In Proceedings of the Sixth Workshop on Uni-
versal Dependencies (UDW, GURT/SyntaxFest 2023),
pp. 47–57, Washington, D.C., March 2023. Associa-
tion for Computational Linguistics. URL https://
aclanthology.org/2023.udw-1.6/.

Hu, X. Meta semantics: Towards better natural lan-
guage understanding and reasoning. arXiv preprint,
abs/2304.10663, 2023. URL https://arxiv.org/
abs/2304.10663.

Kim, N. and Linzen, T. Cogs: A compositional gen-
eralization challenge based on semantic interpretation.
In Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP),
pp. 9087–9105, Online, November 2020. Association
for Computational Linguistics. doi: 10.18653/v1/2020.
emnlp-main.731. URL https://aclanthology.
org/2020.emnlp-main.731/.

Navigli, R., Pinto, M. L., Silvestri, P., Rotondi, D., Cicil-
iano, S., and Scir`e, A. Nounatlas: Filling the gap in nom-
inal semantic role labeling. In Proceedings of the 62nd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pp. 16245–16258,
Bangkok, Thailand, August 2024. Association for Com-
putational Linguistics. doi: 10.18653/v1/2024.acl-long.
857. URL https://aclanthology.org/2024.
acl-long.857/.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S.,
Matena, M., Zhou, Y., Li, W., and Liu, P. J. Explor-
ing the limits of transfer learning with a unified text-to-
text transformer, 2023. URL https://arxiv.org/
abs/1910.10683.

Zhang, S., Zhao, H., and Zhou, J. Semantics-aware in-
ferential network for natural language understanding.
arXiv preprint, abs/2004.13338, 2020. URL https:
//arxiv.org/abs/2004.13338.
```