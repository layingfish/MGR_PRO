import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList


class TigerScoreFusionProcessor(LogitsProcessor):
    def __init__(self, query_vectors, codebooks, identifier_length, num_beams, weight, score_fusion_levels=None, normalize_bias=True):
        self.query_vectors = query_vectors
        self.codebooks = codebooks
        self.identifier_length = int(identifier_length)
        self.num_beams = int(num_beams)
        self.weight = float(weight)
        self.score_fusion_levels = len(codebooks) if score_fusion_levels is None else min(int(score_fusion_levels), len(codebooks))
        self.normalize_bias = bool(normalize_bias)

    def __call__(self, input_ids, scores):
        step = int(input_ids.shape[1]) - 1
        level = step
        if level < 0 or level >= self.score_fusion_levels:
            return scores
        rows = torch.arange(scores.shape[0], device=scores.device) // self.num_beams
        query = self.query_vectors.index_select(0, rows.clamp(max=self.query_vectors.shape[0] - 1)).to(scores.device)
        prefix = torch.zeros_like(query)
        generated = input_ids[:, 1:]
        for previous_level in range(level):
            position = previous_level
            if position < generated.shape[1]:
                codebook = self.codebooks[previous_level].to(scores.device)
                code = generated[:, position].clamp(min=0, max=codebook.shape[0] - 1)
                prefix = prefix + codebook.index_select(0, code)
        codebook = self.codebooks[level].to(scores.device)
        residual = query - prefix
        bias = 2.0 * residual @ codebook.t() - codebook.pow(2).sum(dim=1)
        if self.normalize_bias:
            bias = bias - bias.mean(dim=-1, keepdim=True)
            bias = bias / bias.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        width = min(scores.shape[1], bias.shape[1])
        scores[:, :width] = scores[:, :width] + self.weight * bias[:, :width].to(scores.dtype)
        return scores


class TigerSequenceModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, num_layers, dropout):
        super().__init__()
        config = T5Config(vocab_size=int(vocab_size), d_model=int(hidden_size), num_heads=int(num_heads), num_layers=int(num_layers), num_decoder_layers=int(num_layers), d_ff=1024, dropout_rate=float(dropout), decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
        self.model = T5ForConditionalGeneration(config)

    def forward(self, input_ids, labels):
        attention_mask = (input_ids >= 0).long()
        input_ids = input_ids.clamp_min(0)
        labels = labels.clamp_min(0)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

    @torch.no_grad()
    def encoder_representation(self, input_ids, pooling="last_mean", last_weight=0.7):
        attention_mask = (input_ids >= 0).long()
        input_ids = input_ids.clamp_min(0)
        output = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        mask = attention_mask.bool()
        count = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (output * mask.unsqueeze(-1).float()).sum(dim=1) / count
        last_index = mask.long().sum(dim=1).sub(1).clamp_min(0)
        last = output[torch.arange(output.shape[0], device=output.device), last_index]
        if pooling == "mean":
            return mean
        if pooling == "last":
            return last * (count.squeeze(1) > 0).unsqueeze(-1)
        weight = min(max(float(last_weight), 0.0), 1.0)
        return weight * last + (1.0 - weight) * mean

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, num_beams, num_return_sequences=1, logits_processor=None):
        attention_mask = (input_ids >= 0).long()
        input_ids = input_ids.clamp_min(0)
        processors = LogitsProcessorList([] if logits_processor is None else [logits_processor])
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=int(max_new_tokens), num_beams=int(num_beams), num_return_sequences=int(num_return_sequences), logits_processor=processors)


def identifier_reconstruction(identifier_ids, codebooks, score_fusion_levels=None):
    device = codebooks[0].device
    identifier_ids = identifier_ids.to(device).clamp_min(0)
    output = torch.zeros((identifier_ids.shape[0], codebooks[0].shape[1]), device=device)
    levels = len(codebooks) if score_fusion_levels is None else min(int(score_fusion_levels), len(codebooks))
    for level in range(levels):
        if level < identifier_ids.shape[1]:
            codebook = codebooks[level]
            code = identifier_ids[:, level].clamp(min=0, max=codebook.shape[0] - 1)
            output = output + codebook.index_select(0, code)
    return output


def history_reconstruction(input_ids, codebooks, identifier_length, score_fusion_levels=None):
    usable = input_ids.clamp_min(0)
    if usable.shape[1] < int(identifier_length):
        return torch.zeros((input_ids.shape[0], codebooks[0].shape[1]), device=codebooks[0].device)
    return identifier_reconstruction(usable[:, -int(identifier_length):], codebooks, score_fusion_levels)


@torch.no_grad()
def fit_encoder_bridge(model, loader, codebooks, identifier_length, device, ridge_lambda=1e-3, pooling="last_mean", last_weight=0.7, score_fusion_levels=None):
    sources = []
    targets = []
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        source = model.encoder_representation(input_ids, pooling, last_weight)
        target = identifier_reconstruction(labels, codebooks, score_fusion_levels)
        source = F.normalize(source, dim=-1)
        target = F.normalize(target, dim=-1)
        sources.append(torch.nan_to_num(source).cpu())
        targets.append(torch.nan_to_num(target).cpu())
    x = torch.cat(sources, dim=0).double()
    y = torch.cat(targets, dim=0).double()
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    xtx = x_centered.t().mm(x_centered)
    if float(ridge_lambda) > 0:
        xtx = xtx + float(ridge_lambda) * torch.eye(xtx.shape[0], dtype=xtx.dtype)
    xty = x_centered.t().mm(y_centered)
    try:
        weight = torch.linalg.solve(xtx, xty)
    except RuntimeError:
        weight = torch.linalg.pinv(xtx).mm(xty)
    bias = (y_mean - x_mean.mm(weight)).squeeze(0)
    return weight.float().to(device), bias.float().to(device)
