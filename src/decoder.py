import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, d_model=512, heads=8, n_layers=6, dropout=0.1, max_seq_len=1024):
        super().__init__()

        self.positional_encoding = nn.parameter.Parameter(
            torch.randn(max_seq_len, d_model)
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, heads, dropout) for _ in range(n_layers)]
        )

    def forward(self, x, mask=None):
        x = x + self.positional_encoding[: x.size(1)]
        for layer in self.layers:
            x = layer(x, mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.self_attn = Attention(d_model, heads, dropout, apply_mask=True)
        self.mlp = MultiLayerPerceptron(d_model, d_model * 4, dropout)

    def forward(self, x, mask=None):
        x = self.self_attn(x, mask)
        x = self.mlp(x)

        return x


class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_model, dff, dropout=0.1):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        linear = x + self.linear(x)
        return self.layer_norm(linear)


class Attention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, apply_mask=False):
        super().__init__()
        self.apply_mask = apply_mask
        self.heads = heads
        self.d_k = d_model // heads
        self.scale = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)).item()

        self.keys = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x is of shape (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()

        qry, key, val = self.keys(x).split(d_model, dim=-1)

        # split the input into heads.
        # shape (batch_size, seq_len, heads, d_k)
        qry = qry.reshape(batch_size, seq_len, self.heads, self.d_k)
        key = key.reshape(batch_size, seq_len, self.heads, self.d_k)
        val = val.reshape(batch_size, seq_len, self.heads, self.d_k)

        # transpose to get the shape (batch_size, heads, seq_len, d_k)
        qry = qry.transpose(1, 2)
        key = key.transpose(1, 2)
        val = val.transpose(1, 2)

        # calculate the attention weights. shape (batch_size, heads, seq_len, seq_len)
        A = torch.matmul(qry, key.transpose(-2, -1)) / self.scale

        # apply the mask if possible
        if self.apply_mask:
            # Create causal mask (allowed positions
            causal_mask = (
                torch.tril(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=A.device)
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            if mask is not None:
                # Expand padding mask to (batch, 1, 1, seq_len)
                padding_mask = mask.bool().unsqueeze(1).unsqueeze(2)
                final_mask = causal_mask & padding_mask
            else:
                final_mask = causal_mask
            A = A.masked_fill(final_mask == 0, float("-inf"))

        A = torch.softmax(A, dim=-1)
        A = torch.matmul(A, val)

        # transpose to get the shape (batch_size, seq_len, heads, d_k)
        A = A.transpose(1, 2)

        # concatenate the heads
        A = A.reshape(batch_size, seq_len, d_model)

        A = self.out(A)
        A = self.dropout(A)
        A = A + x
        A = self.norm(A)

        return A
