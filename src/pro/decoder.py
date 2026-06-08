import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration
from .score_fusion import add_geometric_score


class CodeTokenizer:
    def __init__(self, codebook_sizes, modality_tokens=0):
        self.codebook_sizes = [int(x) for x in codebook_sizes]
        self.modality_tokens = int(modality_tokens)
        self.pad_token_id = 0
        self.bos_token_id = 0
        self.eos_token_id = 1
        offset = 2
        self.modality_offset = offset
        offset += self.modality_tokens
        self.level_offsets = []
        for size in self.codebook_sizes:
            self.level_offsets.append(offset)
            offset += size
        self.vocab_size = offset

    def encode_codes(self, codes, modality=None):
        rows = []
        for row_index in range(codes.shape[0]):
            tokens = []
            if self.modality_tokens > 0:
                if modality is None:
                    raise ValueError("modality is required when modality tokens are enabled")
                value = int(modality[row_index])
                if value < 0 or value >= self.modality_tokens:
                    raise ValueError("modality index is outside the configured modality vocabulary")
                tokens.append(self.modality_offset + value)
            for level, code in enumerate(codes[row_index].tolist()):
                tokens.append(self.level_offsets[level] + int(code))
            tokens.append(self.eos_token_id)
            rows.append(tokens)
        return torch.tensor(rows, dtype=torch.long, device=codes.device)

    def level_for_step(self, step):
        if self.modality_tokens > 0:
            if step == 0:
                return None
            return step - 1
        return step

    def token_to_code(self, token, level):
        return int(token) - self.level_offsets[int(level)]

    def allowed_tokens_for_level(self, level):
        offset = self.level_offsets[int(level)]
        return list(range(offset, offset + self.codebook_sizes[int(level)]))


class PrefixTrie:
    def __init__(self):
        self.children = {}
        self.items = {}

    def insert(self, tokens, item_index):
        node = self.children
        prefix = []
        for token in tokens:
            token = int(token)
            prefix.append(token)
            node = node.setdefault(token, {})
        self.items.setdefault(tuple(prefix), []).append(int(item_index))

    def next_tokens(self, prefix):
        node = self.children
        for token in prefix:
            node = node.get(int(token), {})
        return list(node.keys())

    def items_for_prefix(self, prefix):
        return self.items.get(tuple(int(x) for x in prefix), [])


class PrefixRetriever(nn.Module):
    def __init__(self, model_name, embedding_dim, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        if str(model_name) == "t5-small":
            config = T5Config(vocab_size=tokenizer.vocab_size, d_model=512, d_ff=2048, num_layers=6, num_decoder_layers=6, num_heads=8, d_kv=64, dropout_rate=0.1, decoder_start_token_id=tokenizer.pad_token_id, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        else:
            config = T5Config(vocab_size=tokenizer.vocab_size, d_model=512, d_ff=2048, num_layers=6, num_decoder_layers=6, num_heads=8, d_kv=64, dropout_rate=0.1, decoder_start_token_id=tokenizer.pad_token_id, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        self.model = T5ForConditionalGeneration(config)
        self.query_projection = nn.Linear(int(embedding_dim), config.d_model)

    def forward(self, query_embeddings, labels):
        inputs_embeds = self.query_projection(query_embeddings).unsqueeze(1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=query_embeddings.device)
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss

    @torch.no_grad()
    def beam_search(self, query_embedding, trie, quantizer, beam_size, output_size, score_fusion_enabled=False, score_fusion_weight=0.0, score_fusion_normalize_bias=False):
        device = next(self.parameters()).device
        query_embedding = query_embedding.to(device)
        encoder_inputs = self.query_projection(query_embedding.unsqueeze(0)).unsqueeze(1)
        attention_mask = torch.ones((1, 1), dtype=torch.long, device=device)
        encoder_outputs = self.model.encoder(inputs_embeds=encoder_inputs, attention_mask=attention_mask)
        beams = [([], torch.zeros_like(query_embedding), 0.0)]
        total_steps = len(self.tokenizer.codebook_sizes) + (1 if self.tokenizer.modality_tokens > 0 else 0)
        for step in range(total_steps):
            candidates = []
            for prefix, reconstruction, score in beams:
                allowed = trie.next_tokens(prefix)
                if not allowed:
                    continue
                decoder_input = torch.tensor([[self.tokenizer.bos_token_id] + prefix], dtype=torch.long, device=device)
                output = self.model(encoder_outputs=encoder_outputs, attention_mask=attention_mask, decoder_input_ids=decoder_input, use_cache=False)
                logits = F.log_softmax(output.logits[0, -1], dim=-1)
                level = self.tokenizer.level_for_step(step)
                if score_fusion_enabled and level is not None:
                    mapping = {token: self.tokenizer.token_to_code(token, level) for token in allowed}
                    logits = add_geometric_score(logits, allowed, mapping, query_embedding, reconstruction, quantizer.codebooks[int(level)], score_fusion_weight, score_fusion_normalize_bias)
                for token in allowed:
                    next_reconstruction = reconstruction
                    if level is not None:
                        code = self.tokenizer.token_to_code(token, level)
                        next_reconstruction = reconstruction + quantizer.codebooks[int(level)][int(code)].to(device)
                    candidates.append((prefix + [int(token)], next_reconstruction, score + float(logits[int(token)].item())))
            candidates.sort(key=lambda x: x[2], reverse=True)
            beams = candidates[:int(beam_size)]
        results = []
        for prefix, _, score in beams:
            for item in trie.items_for_prefix(prefix):
                results.append((item, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:int(output_size)]
