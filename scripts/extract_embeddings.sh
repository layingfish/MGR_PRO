set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python -m pro.extract_embeddings --config configs/pro_multimodal.yaml
