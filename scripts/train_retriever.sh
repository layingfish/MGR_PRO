set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python -m pro.train_retriever --config configs/pro_multimodal.yaml
