# Notebooks Breakdown

## 1. Data Import & Review
 - **Goal:** Load & explore COGS, PropBank, UD datasets
 - **Checkpoint 1A:** Dataset format validation
 - **Checkpoint 1B:** Exploratory statistics & visualizations

## 2. Baseline Metrics
 - **Goal:** Evaluate T5, BART, GPT-4o-mini on original datasets
 - **Checkpoint 2A:** Exact Match & F1 score for each model
 - **Checkpoint 2B:** Analyze failure cases in compositional structures

 ## 3. Preprocessings (SRL & Knowledge Graphs)
 - **Goal:** Enhance datasets with SRL labels & knowledge graphs
 - **Checkpoint 3A:** Run SRL annotation pipeline (Agent, Theme, Recipient)
 - **Checkpoint 3B:** Convert text into knowledge graphs

 ## 4. Train Meta-Semantic Model
 - **Goal:** Train a unified model with SRL-aware inputs, KG embeddings, & rule-based validation
 - **Checkpoint 4A:** Train baseline Transformer with SRL
 - **Checkpoint 4B:** Train Graph-Enhanced Transformer
 - **Checkpoint 4C:** Train Hybrid Neural-Symbolic model

 ## 5. Evaluate Generalization
 - **Goal:** Compare meta-semantic models against baseline LLMs
 - **Checkpoint 5A:** Compute Compositional Generalization Score
 - **Checkpoint 5B:** Perform Ablation Study to isolate contributions of SRL, KG, and rule constraints
 - **Checkpoint 5C:** Visualize error analysis and qualitative results