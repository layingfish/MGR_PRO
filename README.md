# Overview

This repository contains the implementation for Prefix Retention Optimization. The `src/pro` package contains the CLIP-SF embedding extraction wrapper and the three method components described in the paper: prefix ranking distillation, vocabulary scheduling, and geometric score fusion. The `baseline/genius` and `baseline/tiger` directories contain the comparison pipelines used for the GENIUS and TIGER experiments. Place data manifests, encoder checkpoints, qrels, TIGER semantic identifier sequences, and the external M-BEIR/CLIP-SF reference code root at local paths and point the config files to those paths.

# Environment Setup

```bash
conda env create -f environment.yaml
conda activate pro
```

# Running Commands

```bash
bash scripts/extract_embeddings.sh
bash scripts/train_quantizer.sh
bash scripts/train_retriever.sh
bash scripts/evaluate.sh
bash scripts/run_full_pipeline.sh
```
