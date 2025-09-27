import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        # Precomputed constant: sqrt(2/π) ≈ 0.79788456
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


if __name__ == "__main__":
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, 768)
    out = ffn(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Success! Shape preserved as expected.")

