import numpy as np
import torch
import torch.nn as nn

from torch import (
    Tensor,
    LongTensor,
)


__all__ = (
    'TokenEmbedding',
    'PositionalEmbedding',
    'TemporalEmbedding',
)


class TokenEmbedding(nn.Embedding):

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 64,
                 ):

        # params
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # (V x d)
        super().__init__(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )


class PositionalEmbedding(nn.Module):

    def __init__(self,
                 sequence_len: int,
                 embedding_dim: int = 64,
                 ):
        super().__init__()

        # params
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim

        # layers
        self.embedding = nn.Embedding(
            num_embeddings=sequence_len,
            embedding_dim=embedding_dim,
        )

    def forward(self, tokens: LongTensor):

        b = tokens.size(0)

        # [process]
        # (1) unsqueeze: (L x d) -> (1 x L x d)
        # (2) repeat: (1 x L x d) -> (b x L x d)
        x = self.embedding.weight.unsqueeze(0).repeat(b, 1, 1)

        return x


class TemporalEmbedding(nn.Module):

    def __init__(self, embedding_dim: int = 16):
        super().__init__()

        # params
        self.embedding_dim = embedding_dim

        # init (see TGSRec)
        temporal_init = torch.from_numpy(1 / 10 ** np.linspace(0, 9, embedding_dim))

        # layers
        self.weight = nn.Parameter(temporal_init.float())
        self.bias = nn.Parameter(torch.zeros(embedding_dim).float())

    def forward(self, stamps: Tensor):

        # (d) -> (1 x 1 x d)
        weight = self.weight.view(1, 1, -1)
        bias = self.bias.view(1, 1, -1)

        # (b x L) -> (b x L x 1)
        stamps = stamps.unsqueeze(-1)

        # (b x L x 1) times (1 x 1 x d) -> (b x L x d)
        span = stamps * weight + bias
        harmonic = torch.cos(span)

        return harmonic
