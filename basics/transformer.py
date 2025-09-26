import tiktoken
import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 32,  # smaller for testing speed
    "emb_dim": 64,         # smaller for testing speed
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False
}


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('coefficient', torch.tensor(0.79788456))

    def forward(self, x):
        coeff = self.coefficient.to(x)
        return 0.5 * x * (1 + torch.tanh(coeff * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads,
                         self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads,
                               self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads,
                             self.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, float('-inf'))
        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean)/torch.sqrt(var+self.eps)
        return self.scale*norm_x+self.shift


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_ids = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_ids)[None, :, :]
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def pad_to_same_length(batch, pad_token_id=0):
    max_len = max(len(x) for x in batch)
    padded = [torch.cat([x, torch.full((max_len - len(x),), pad_token_id,
                        dtype=torch.long)]) if len(x) < max_len else x for x in batch]
    batch_tensor = torch.stack(padded, dim=0)
    return batch_tensor


if __name__ == "__main__":
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch = [torch.tensor(tokenizer.encode(txt1)),
             torch.tensor(tokenizer.encode(txt2))]
    batch_tensor = pad_to_same_length(batch)
    print("Batch tensor:\n", batch_tensor)
    model = GPTModel(GPT_CONFIG_124M)
    out = model(batch_tensor)
    print("\nInput batch:\n", batch_tensor)
    print("\nOutput shape:", out.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: ", total_params)
    print("Token embedding layer shape: ", model.tok_emb.weight.shape)
    print("Output layer shape: ", model.out_head.weight.shape)

    total_params_gpt2 = (total_params - sum(p.numel()
                         for p in model.out_head.parameters()))

    def generate_text_simple(model, idx, max_new_tokens, context_size):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    start_context = "Hello, I am doing"
    encoded = tokenizer.encode(start_context)
    print("encoded: ", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("Encoded_tensor.shape: ", encoded_tensor.shape)
    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output: ", out)
    print("Length of the output: ", len(out[0]))

    decoded_txt = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_txt)
