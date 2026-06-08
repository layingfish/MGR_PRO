import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from tiger import TigerSequenceModel
import sys
sys.path.insert(0, "src")
from pro.config import load_config, require_path, set_seed


class TigerIdentifierSequenceDataset(Dataset):
    def __init__(self, path, sequence_length, identifier_length):
        self.rows = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                values = [int(x) for x in line.strip().replace(",", " ").split() if x]
                if len(values) > int(identifier_length):
                    values = values[-int(sequence_length):]
                    self.rows.append(values)
        self.identifier_length = int(identifier_length)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        values = self.rows[index]
        return torch.tensor(values[:-self.identifier_length], dtype=torch.long), torch.tensor(values[-self.identifier_length:], dtype=torch.long)


def collate(rows):
    histories, labels = zip(*rows)
    max_history = max(x.numel() for x in histories)
    inputs = torch.full((len(rows), max_history), -1, dtype=torch.long)
    for i, value in enumerate(histories):
        inputs[i, -value.numel():] = value
    labels = torch.stack(labels)
    return inputs, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiger_recommendation.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    identifier_length = int(config["tiger"]["decoder_tokens"])
    dataset = TigerIdentifierSequenceDataset(require_path(config["paths"]["train_sequences"], "train_sequences"), config["tiger"]["sequence_length"], identifier_length)
    loader = DataLoader(dataset, batch_size=int(config["tiger"]["batch_size"]), shuffle=True, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TigerSequenceModel(config["tiger"]["transformer_vocab_size"], config["tiger"]["hidden_size"], config["tiger"]["num_heads"], config["tiger"]["num_layers"], config["tiger"]["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["tiger"]["learning_rate"]))
    steps = 0
    while steps < int(config["tiger"]["max_steps"]):
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss = model(input_ids, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps >= int(config["tiger"]["max_steps"]):
                break
    output = require_path(config["paths"]["recommender_checkpoint"], "recommender_checkpoint")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config["tiger"]}, output)


if __name__ == "__main__":
    main()
