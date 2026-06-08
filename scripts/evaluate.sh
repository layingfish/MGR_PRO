set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python -m pro.evaluate --config configs/pro_multimodal.yaml
