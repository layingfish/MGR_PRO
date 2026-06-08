import argparse
import sys
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from .config import load_config, require_path, is_placeholder, set_seed
from .data import hash_item_id, hash_query_id


def _load_external_modules(reference_code_root):
    root = Path(reference_code_root)
    for path in (root / "src", root):
        path = str(path)
        if path not in sys.path:
            sys.path.insert(0, path)
    from data.mbeir_dataset import MBEIRMainDataset, MBEIRCandidatePoolDataset, Mode
    from data.preprocessing.utils import format_string
    from models.uniir_clip.clip_nofusion.clip_nf import CLIPNoFusion
    return MBEIRMainDataset, MBEIRCandidatePoolDataset, Mode, format_string, CLIPNoFusion


def _parse_size(value):
    if isinstance(value, str):
        parts = [int(x.strip()) for x in value.replace("x", ",").split(",") if x.strip()]
    else:
        parts = [int(x) for x in value]
    if len(parts) != 2:
        raise ValueError("image_size must contain two integers")
    return parts[0], parts[1]


def _black_image(size):
    width, height = size
    return Image.new("RGB", (int(width), int(height)), (0, 0, 0))


def _make_query_loader(modules, data_root, query_path, pool_path, query_instruct_path, enable_query_instruct, preprocess, batch_size, workers, image_size):
    MBEIRMainDataset, _, Mode, format_string, _ = modules
    dataset = MBEIRMainDataset(
        mbeir_data_dir=data_root,
        query_data_path=query_path,
        cand_pool_path=pool_path,
        query_instruct_path=query_instruct_path,
        img_preprocess_fn=preprocess,
        mode=Mode.TRAIN,
        enable_query_instruct=bool(enable_query_instruct),
        shuffle_cand=False,
        hard_neg_num=0,
        returns=None,
        print_config=True,
    )
    black = _black_image(image_size)

    def collate(rows):
        ids, texts, images, text_masks, image_masks = [], [], [], [], []
        for row in rows:
            ids.append(hash_query_id(row["qid"]))
            query = row["query"]
            text = format_string((query.get("txt") or "").strip())
            image = query.get("img", None)
            texts.append(text if text else "")
            text_masks.append(1 if text else 0)
            if image is None:
                images.append(preprocess(black))
                image_masks.append(0)
            else:
                images.append(image)
                image_masks.append(1)
        return {
            "ids": ids,
            "texts": texts,
            "images": torch.stack(images, dim=0),
            "text_mask": torch.tensor(text_masks, dtype=torch.long),
            "img_mask": torch.tensor(image_masks, dtype=torch.long),
        }

    return DataLoader(dataset, batch_size=int(batch_size), num_workers=int(workers), shuffle=False, collate_fn=collate, drop_last=False)


def _make_pool_loader(modules, data_root, pool_path, preprocess, batch_size, workers, image_size):
    _, MBEIRCandidatePoolDataset, _, format_string, _ = modules
    dataset = MBEIRCandidatePoolDataset(
        mbeir_data_dir=data_root,
        cand_pool_data_path=pool_path,
        img_preprocess_fn=preprocess,
        returns=None,
        print_config=True,
    )
    black = _black_image(image_size)

    def collate(rows):
        ids, texts, images, text_masks, image_masks = [], [], [], [], []
        for row in rows:
            ids.append(hash_item_id(row["did"]))
            text = format_string((row.get("txt") or "").strip())
            image = row.get("img", None)
            texts.append(text if text else "")
            text_masks.append(1 if text else 0)
            if image is None:
                images.append(preprocess(black))
                image_masks.append(0)
            else:
                images.append(image)
                image_masks.append(1)
        return {
            "ids": ids,
            "texts": texts,
            "images": torch.stack(images, dim=0),
            "text_mask": torch.tensor(text_masks, dtype=torch.long),
            "img_mask": torch.tensor(image_masks, dtype=torch.long),
        }

    return DataLoader(dataset, batch_size=int(batch_size), num_workers=int(workers), shuffle=False, collate_fn=collate, drop_last=False)


@torch.no_grad()
def _encode_loader(loader, model, tokenizer, device, output_path):
    id_to_index = {}
    image_features = []
    text_features = []
    image_masks = []
    text_masks = []
    index = 0
    for batch in tqdm(loader):
        ids = batch["ids"]
        images = batch["images"].to(device)
        tokens = tokenizer(batch["texts"]).to(device)
        image_embedding, text_embedding = model.encode_multimodal_input(images, tokens)
        image_embedding = image_embedding.float().cpu()
        text_embedding = text_embedding.float().cpu()
        for row, item_id in enumerate(ids):
            item_id = int(item_id)
            if item_id in id_to_index:
                continue
            id_to_index[item_id] = index
            image_features.append(image_embedding[row])
            text_features.append(text_embedding[row])
            image_masks.append(batch["img_mask"][row].cpu())
            text_masks.append(batch["text_mask"][row].cpu())
            index += 1
    result = {
        "img": torch.stack(image_features, dim=0),
        "text": torch.stack(text_features, dim=0),
        "img_mask": torch.stack(image_masks, dim=0),
        "text_mask": torch.stack(text_masks, dim=0),
        "id_to_index": id_to_index,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)


def _load_encoder(modules, extract_config, device):
    _, _, _, _, CLIPNoFusion = modules
    root = Path(require_path(extract_config.get("genir_root"), "extract.genir_root"))
    checkpoint = extract_config.get("checkpoint_path")
    if checkpoint is None or is_placeholder(checkpoint):
        checkpoint = root / "checkpoint" / "CLIP_SF" / "clip_sf_large.pth"
    else:
        checkpoint = Path(checkpoint)
    clip_root = extract_config.get("clip_model_root")
    if clip_root is None or is_placeholder(clip_root):
        clip_root = root / "checkpoint" / "CLIP"
    model = CLIPNoFusion(
        model_name=extract_config.get("model_name", "ViT-L/14"),
        download_root=str(clip_root),
        config=None,
    ).to(device)
    model.float().eval()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=False)
    return model, model.get_img_preprocess_fn(), model.get_tokenizer()


def _extract_split(split, config, modules, model, preprocess, tokenizer, device):
    paths = config["paths"]
    extract_config = config["extract"]
    query_manifest = paths.get(f"{split}_query_manifest")
    item_manifest = paths.get(f"{split}_item_manifest")
    query_output = paths.get(f"{split}_query_embeddings")
    item_output = paths.get(f"{split}_item_embeddings")
    values = [query_manifest, item_manifest, query_output, item_output]
    if any(value is None or is_placeholder(value) for value in values):
        return
    data_root = require_path(extract_config.get("mbeir_data_root"), "extract.mbeir_data_root")
    image_size = _parse_size(extract_config.get("image_size", "224,224"))
    batch_size = int(extract_config.get("batch_size", 64))
    workers = int(extract_config.get("num_workers", 8))
    query_instruct_path = extract_config.get("query_instruct_path", "instructions/query_instructions.tsv")
    enable_query_instruct = bool(extract_config.get("enable_query_instruct", True))
    query_loader = _make_query_loader(modules, data_root, query_manifest, item_manifest, query_instruct_path, enable_query_instruct, preprocess, batch_size, workers, image_size)
    pool_loader = _make_pool_loader(modules, data_root, item_manifest, preprocess, batch_size, workers, image_size)
    _encode_loader(query_loader, model, tokenizer, device, query_output)
    _encode_loader(pool_loader, model, tokenizer, device, item_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 2023))
    extract_config = config["extract"]
    device = torch.device(extract_config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    modules = _load_external_modules(require_path(extract_config.get("reference_code_root"), "extract.reference_code_root"))
    model, preprocess, tokenizer = _load_encoder(modules, extract_config, device)
    for split in ("train", "validation", "test"):
        _extract_split(split, config, modules, model, preprocess, tokenizer, device)


if __name__ == "__main__":
    main()
