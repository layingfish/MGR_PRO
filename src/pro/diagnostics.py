import argparse
import json
from pathlib import Path
import torch
from .config import load_config, require_path, set_seed
from .data import load_torch_file, load_tensor, load_qrels
from .quantizer import ResidualQuantizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 2023))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = config["paths"]
    checkpoint = load_torch_file(require_path(paths["quantizer_checkpoint"], "quantizer_checkpoint"), map_location="cpu")
    quantizer_config = checkpoint["config"]
    quantizer = ResidualQuantizer(quantizer_config["embedding_dim"], quantizer_config["codebook_sizes"]).to(device)
    quantizer.load_state_dict(checkpoint["state_dict"])
    quantizer.eval()
    queries = load_tensor(require_path(paths["test_query_embeddings"], "test_query_embeddings")).to(device)
    items = load_tensor(require_path(paths["test_item_embeddings"], "test_item_embeddings")).to(device)
    qrels = load_qrels(require_path(paths["test_qrels"], "test_qrels"))
    beam_size = int(config.get("diagnostics", {}).get("beam_size", 20))
    with torch.no_grad():
        codes, _, prefixes = quantizer.encode(items)
    rates = []
    for level, prefix in enumerate(prefixes):
        hits = 0
        total = 0
        scores = queries @ prefix.t()
        top_items = torch.topk(scores, k=min(beam_size, scores.shape[1]), dim=1).indices.cpu()
        for qid, relevant in qrels.items():
            target = next(iter(relevant))
            if int(target) in set(top_items[int(qid)].tolist()):
                hits += 1
            total += 1
        rates.append({"level": level + 1, "survival_rate": hits / max(total, 1)})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(rates, handle, indent=2)


if __name__ == "__main__":
    main()
