import argparse
import json
from pathlib import Path
import torch
from .config import load_config, require_path, set_seed, is_placeholder
from .data import load_torch_file, load_feature_store, load_qrels, load_modalities
from .quantizer import ResidualQuantizer
from .decoder import CodeTokenizer, PrefixRetriever, PrefixTrie


def recall_at(results, qrels, k):
    hits = 0
    total = 0
    for qid, relevant in qrels.items():
        predicted = [item for item, _ in results.get(qid, [])[:k]]
        hits += int(any(item in relevant for item in predicted))
        total += 1
    return hits / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 2023))
    paths = config["paths"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantizer_checkpoint = load_torch_file(require_path(paths["quantizer_checkpoint"], "quantizer_checkpoint"), map_location="cpu")
    quantizer_config = quantizer_checkpoint["config"]
    quantizer = ResidualQuantizer(
        quantizer_config["embedding_dim"],
        quantizer_config["codebook_sizes"],
        quantizer_config.get("combiner_projection_dim", 2560),
        quantizer_config.get("combiner_hidden_dim", 5120),
        quantizer_config.get("kmeans_init", True),
        quantizer_config.get("kmeans_iters", 1000),
        quantizer_config.get("codebook_decay", 0.9),
        quantizer_config.get("ema_update", True),
        quantizer_config.get("threshold_ema_dead_code", 2),
    ).to(device)
    quantizer.load_state_dict(quantizer_checkpoint["state_dict"])
    quantizer.eval()
    tokenizer = CodeTokenizer(quantizer_config["codebook_sizes"], quantizer_config.get("modality_tokens", 0))
    retriever_checkpoint = load_torch_file(require_path(paths["retriever_checkpoint"], "retriever_checkpoint"), map_location="cpu")
    retriever = PrefixRetriever(config["retriever"]["model_name"], config["retriever"]["embedding_dim"], tokenizer).to(device)
    retriever.load_state_dict(retriever_checkpoint["state_dict"])
    retriever.eval()
    queries = load_feature_store(require_path(paths["test_query_embeddings"], "test_query_embeddings"))
    items = load_feature_store(require_path(paths["test_item_embeddings"], "test_item_embeddings"))
    qrels = load_qrels(require_path(paths["test_qrels"], "test_qrels"))
    modality_path = paths.get("test_item_modalities")
    if modality_path is None or is_placeholder(modality_path):
        modality_path = paths.get("test_item_manifest")
    if modality_path is not None and is_placeholder(modality_path):
        modality_path = None
    item_modalities = None if modality_path is None else load_modalities(modality_path, len(items))
    with torch.no_grad():
        codes, _, _ = quantizer.encode(items.all())
        if item_modalities is not None:
            item_modalities = item_modalities.to(device)
        tokens = tokenizer.encode_codes(codes, item_modalities if tokenizer.modality_tokens > 0 else None).cpu().tolist()
    trie = PrefixTrie()
    for item_index, token_row in enumerate(tokens):
        trie.insert(token_row[:-1], int(items.ids[item_index]))
    results = {}
    for qid in qrels.keys():
        with torch.no_grad():
            query = quantizer.encode_features(queries.get(int(qid))).squeeze(0)
        results[int(qid)] = retriever.beam_search(query, trie, quantizer, config["retriever"]["beam_size"], config["retriever"]["output_size"], config["retriever"].get("score_fusion_enabled", False), config["retriever"].get("score_fusion_weight", 0.0), config["retriever"].get("score_fusion_normalize_bias", False))
    metrics = {"R@1": recall_at(results, qrels, 1), "R@5": recall_at(results, qrels, 5)}
    output = require_path(paths["output_json"], "output_json")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w", encoding="utf-8") as handle:
        json.dump({"metrics": metrics, "results": {str(k): v for k, v in results.items()}}, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
