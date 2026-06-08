set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python -m pro.train_quantizer --config configs/pro_multimodal.yaml
