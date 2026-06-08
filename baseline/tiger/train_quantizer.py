import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.insert(0, "src")
from pro.config import load_config, require_path, set_seed
from pro.data import load_torch_file
from pro.quantizer import ResidualQuantizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiger_recommendation.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    embeddings = load_torch_file(require_path(config["paths"]["item_embeddings"], "item_embeddings"), map_location="cpu").float()
    dataset = TensorDataset(embeddings)
    loader = DataLoader(dataset, batch_size=int(config["tiger"]["batch_size"]), shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantizer = ResidualQuantizer(
        config["tiger"]["item_embedding_dim"],
        config["tiger"]["codebook_sizes"],
        config["tiger"].get("combiner_projection_dim", 2560),
        config["tiger"].get("combiner_hidden_dim", 5120),
        config["tiger"].get("kmeans_init", True),
        config["tiger"].get("kmeans_iters", 1000),
        config["tiger"].get("codebook_decay", 0.9),
        config["tiger"].get("ema_update", True),
        config["tiger"].get("threshold_ema_dead_code", 2),
    ).to(device)
    optimizer = torch.optim.AdamW(quantizer.parameters(), lr=float(config["tiger"]["learning_rate"]))
    steps = 0
    while steps < int(config["tiger"]["max_steps"]):
        for (batch,) in loader:
            batch = batch.to(device)
            _, reconstruction, _, rq_loss = quantizer(batch)
            loss = rq_loss + torch.nn.functional.mse_loss(reconstruction, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps >= int(config["tiger"]["max_steps"]):
                break
    output = require_path(config["paths"]["quantizer_checkpoint"], "quantizer_checkpoint")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": quantizer.state_dict(), "config": {"embedding_dim": config["tiger"]["item_embedding_dim"], "codebook_sizes": config["tiger"]["codebook_sizes"], "kmeans_init": config["tiger"].get("kmeans_init", True), "kmeans_iters": config["tiger"].get("kmeans_iters", 1000), "codebook_decay": config["tiger"].get("codebook_decay", 0.9), "ema_update": config["tiger"].get("ema_update", True), "threshold_ema_dead_code": config["tiger"].get("threshold_ema_dead_code", 2)}}, output)


if __name__ == "__main__":
    main()
