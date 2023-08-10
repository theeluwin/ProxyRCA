import torch.nn as nn

from typing import Optional

from torch import LongTensor

from tools.utils import fix_random_seed

from ..layers import (
    TokenEmbedding,
    PositionalEmbedding,
)


__all__ = (
    'VanillaItemEncoder',
)


class VanillaItemEncoder(nn.Module):

    def __init__(self,
                 num_items: int,
                 sequence_len: int,
                 hidden_dim: int = 64,
                 random_seed: Optional[int] = None,
                 ):
        super().__init__()

        # data params
        self.num_items = num_items
        self.sequence_len = sequence_len

        # main params
        self.hidden_dim = hidden_dim

        # optional params
        self.random_seed = random_seed

        # set seed
        if random_seed is not None:
            fix_random_seed(random_seed)

        # 0: padding token
        # 1 ~ V: item tokens
        self.vocab_size = num_items + 1

        # embedding layers
        self.token_embedding = TokenEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=hidden_dim,
        )
        self.positional_embedding = PositionalEmbedding(
            sequence_len=sequence_len,
            embedding_dim=hidden_dim,
        )

    def forward(self,
                tokens: LongTensor,  # (b x L|C)
                positional: bool = True,
                ):

        # get item vector
        vector = self.token_embedding(tokens)

        # optionals
        if positional:
            vector = vector + self.positional_embedding(tokens)

        return vector
