import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tiger import TigerSequenceModel, TigerScoreFusionProcessor, fit_encoder_bridge, history_reconstruction
from train_recommender import TigerIdentifierSequenceDataset, collate
import sys
sys.path.insert(0, "src")
from pro.config import load_config, require_path, set_seed
from pro.data import load_torch_file
from pro.quantizer import ResidualQuantizer


def sequence_hit(predictions, labels, k):
    batch = labels.shape[0]
    predictions = predictions[:, -labels.shape[1]:].reshape(batch, -1, labels.shape[1])
    top_predictions = predictions[:, :int(k), :]
    return (top_predictions == labels[:, None, :]).all(dim=2).any(dim=1).sum().item()


def sequence_ndcg(predictions, labels, k):
    batch = labels.shape[0]
    predictions = predictions[:, -labels.shape[1]:].reshape(batch, -1, labels.shape[1])
    top_predictions = predictions[:, :int(k), :]
    matches = (top_predictions == labels[:, None, :]).all(dim=2)
    ranks = torch.arange(1, top_predictions.shape[1] + 1, dtype=torch.float32, device=matches.device)
    discounts = 1.0 / torch.log2(ranks + 1.0)
    return (matches.float() * discounts.unsqueeze(0)).max(dim=1).values.sum().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiger_recommendation.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_torch_file(require_path(config["paths"]["recommender_checkpoint"], "recommender_checkpoint"), map_location="cpu")
    model_config = checkpoint["config"]
    model = TigerSequenceModel(model_config["transformer_vocab_size"], model_config["hidden_size"], model_config["num_heads"], model_config["num_layers"], model_config["dropout"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    identifier_length = int(config["tiger"]["decoder_tokens"])
    quantizer = None
    if bool(config["tiger"].get("score_fusion_enabled", False)):
        quantizer_checkpoint = load_torch_file(require_path(config["paths"]["quantizer_checkpoint"], "quantizer_checkpoint"), map_location="cpu")
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
    dataset = TigerIdentifierSequenceDataset(require_path(config["paths"]["test_sequences"], "test_sequences"), config["tiger"]["sequence_length"], identifier_length)
    loader = DataLoader(dataset, batch_size=int(config["tiger"]["batch_size"]), shuffle=False, collate_fn=collate)
    bridge = None
    if quantizer is not None and str(config["tiger"].get("score_fusion_query_source", "history_reconstruction")).lower() == "encoder_bridge":
        bridge_dataset = TigerIdentifierSequenceDataset(require_path(config["paths"]["train_sequences"], "train_sequences"), config["tiger"]["sequence_length"], identifier_length)
        bridge_loader = DataLoader(bridge_dataset, batch_size=int(config["tiger"].get("score_fusion_bridge_batch_size", config["tiger"]["batch_size"])), shuffle=False, collate_fn=collate)
        bridge = fit_encoder_bridge(model, bridge_loader, [x.detach().to(device) for x in quantizer.codebooks], identifier_length, device, config["tiger"].get("score_fusion_bridge_ridge_lambda", 1e-3), config["tiger"].get("score_fusion_encoder_pooling", "last_mean"), config["tiger"].get("score_fusion_last_weight", 0.7), config["tiger"].get("score_fusion_levels", None))
    total = 0
    hit5 = 0
    hit10 = 0
    ndcg5 = 0.0
    ndcg10 = 0.0
    returns = min(int(config["tiger"]["beam_size"]), 10)
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            processor = None
            if quantizer is not None:
                codebooks = [x.detach().to(device) for x in quantizer.codebooks]
                if bridge is None:
                    query_vectors = history_reconstruction(input_ids, codebooks, identifier_length, config["tiger"].get("score_fusion_levels", None))
                else:
                    source = model.encoder_representation(input_ids, config["tiger"].get("score_fusion_encoder_pooling", "last_mean"), config["tiger"].get("score_fusion_last_weight", 0.7))
                    source = torch.nan_to_num(F.normalize(source, dim=-1))
                    query_vectors = torch.nan_to_num(F.normalize(source @ bridge[0] + bridge[1].unsqueeze(0), dim=-1))
                processor = TigerScoreFusionProcessor(query_vectors, codebooks, identifier_length, config["tiger"]["beam_size"], config["tiger"].get("score_fusion_weight", 0.0), config["tiger"].get("score_fusion_levels", None), config["tiger"].get("score_fusion_normalize_bias", True))
            generated = model.generate(input_ids, identifier_length, config["tiger"]["beam_size"], returns, processor).cpu()
            labels_cpu = labels.cpu()
            hit5 += sequence_hit(generated, labels_cpu, min(5, returns))
            hit10 += sequence_hit(generated, labels_cpu, min(10, returns))
            ndcg5 += sequence_ndcg(generated, labels_cpu, min(5, returns))
            ndcg10 += sequence_ndcg(generated, labels_cpu, min(10, returns))
            total += int(labels_cpu.shape[0])
    metrics = {"R@5": hit5 / max(total, 1), "R@10": hit10 / max(total, 1), "N@5": ndcg5 / max(total, 1), "N@10": ndcg10 / max(total, 1)}
    output = require_path(config["paths"]["output_json"], "output_json")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
