set -euo pipefail
bash scripts/extract_embeddings.sh
bash scripts/train_quantizer.sh
bash scripts/train_retriever.sh
bash scripts/evaluate.sh
