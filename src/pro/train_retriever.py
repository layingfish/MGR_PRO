import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .config import load_config, require_path, set_seed, is_placeholder
from .data import load_torch_file, load_feature_store, load_pairs, load_modalities, PairEmbeddingDataset
from .quantizer import ResidualQuantizer
from .decoder import CodeTokenizer, PrefixRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 2023))
    paths = config["paths"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_torch_file(require_path(paths["quantizer_checkpoint"], "quantizer_checkpoint"), map_location="cpu")
    quantizer_config = checkpoint["config"]
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
    quantizer.load_state_dict(checkpoint["state_dict"])
    quantizer.eval()
    tokenizer = CodeTokenizer(quantizer_config["codebook_sizes"], quantizer_config.get("modality_tokens", 0))
    retriever = PrefixRetriever(config["retriever"]["model_name"], config["retriever"]["embedding_dim"], tokenizer).to(device)
    q = load_feature_store(require_path(paths["train_query_embeddings"], "train_query_embeddings"))
    x = load_feature_store(require_path(paths["train_item_embeddings"], "train_item_embeddings"))
    pair_path = paths.get("train_pairs")
    if pair_path is None or is_placeholder(pair_path):
        pair_path = paths.get("train_query_manifest")
    if pair_path is not None and is_placeholder(pair_path):
        pair_path = None
    if pair_path is None and (q.id_to_index is not None or x.id_to_index is not None):
        raise ValueError("Set train_pairs or train_query_manifest when using embedding dictionaries with id_to_index")
    pairs = load_pairs(pair_path, min(len(q), len(x)))
    modality_path = paths.get("train_item_modalities")
    if modality_path is None or is_placeholder(modality_path):
        modality_path = paths.get("train_item_manifest")
    if modality_path is not None and is_placeholder(modality_path):
        modality_path = None
    item_modalities = None if modality_path is None else load_modalities(modality_path, len(x))
    dataset = PairEmbeddingDataset(q, x, pairs, item_modalities)
    loader = DataLoader(dataset, batch_size=int(config["retriever"]["batch_size"]), shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(retriever.parameters(), lr=float(config["retriever"]["learning_rate"]))
    for _ in range(int(config["retriever"]["epochs"])):
        retriever.train()
        for query, item, _, modality in loader:
            modality = modality.to(device)
            with torch.no_grad():
                codes, _, _ = quantizer.encode(item)
                query_embedding = quantizer.encode_features(query)
                labels = tokenizer.encode_codes(codes, modality if tokenizer.modality_tokens > 0 else None)
            loss = retriever(query_embedding, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    output = require_path(paths["retriever_checkpoint"], "retriever_checkpoint")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": retriever.state_dict(), "config": config["retriever"], "tokenizer": {"codebook_sizes": tokenizer.codebook_sizes, "modality_tokens": tokenizer.modality_tokens}}, output)


if __name__ == "__main__":
    main()
