import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, d_model=512, heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        """
        This decoder will receive pre-trained embeddings from the CLIP model.
        Hence, we are not treating x as a tensor of indices but as actual embeddings
        in the correct shape.

        d_model has to be 512, as the CLIP model outputs 512-dim embeddings.
        """

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, heads, dropout) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.self_attn = Attention(d_model, heads, dropout, apply_mask=True)
        self.mlp = MultiLayerPerceptron(d_model, d_model * 4, dropout)

    def forward(self, x):
        x = self.self_attn(x)
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
        self.scale = torch.sqrt(torch.tensor(self.d_k)).item()

        self.keys = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is of shape (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()

        qry, key, val = self.keys(x).split(d_model, dim=-1)

        # split the input into heads.
        # qry is of shape (batch_size, seq_len, heads, d_k)
        qry = qry.reshape(batch_size, seq_len, self.heads, self.d_k)
        key = key.reshape(batch_size, seq_len, self.heads, self.d_k)
        val = val.reshape(batch_size, seq_len, self.heads, self.d_k)

        # transpose to get the shape (batch_size, heads, seq_len, d_k)
        qry = qry.transpose(1, 2)
        key = key.transpose(1, 2)
        val = val.transpose(1, 2)

        # calculate the attention weights
        A = torch.matmul(qry, key.transpose(-2, -1)) / self.scale

        # apply the mask if possible
        if self.apply_mask:
            mask = torch.tril(torch.ones(A.shape[-2:], device=A.device))
            mask = mask.unsqueeze(0)  # add batch dimension
            A = A.masked_fill(mask == 0, float("-inf"))

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
