import numpy as np
import torch.nn as nn

from typing import Optional

from torch import (
    cat as torch_cat,
    Tensor,
    LongTensor,
)
from torch.nn.functional import normalize as F_normalize

from tools.utils import fix_random_seed

from ..layers import TokenEmbedding


__all__ = (
    'MixdimItemEncoder',
)


class MixdimItemEncoder(nn.Module):

    def __init__(self,
                 num_items: int,
                 ifeatures: np.ndarray,
                 ifeature_dim: int,
                 icontext_dim: int,
                 hidden_dim: int = 256,
                 sparse_dim: int = 128,
                 num_dense_item: int = 1024,
                 dropout_prob: float = 0.1,
                 random_seed: Optional[int] = None,
                 ):
        super().__init__()

        # data params
        self.num_items = num_items
        self.ifeature_dim = ifeature_dim
        self.icontext_dim = icontext_dim

        # main params
        self.hidden_dim = hidden_dim
        self.sparse_dim = sparse_dim
        self.num_dense_item = num_dense_item

        # optional params
        self.dropout_prob = dropout_prob
        self.random_seed = random_seed

        # set seed
        if random_seed is not None:
            fix_random_seed(random_seed)

        # ifeature cache
        self.ifeature_cache = nn.Embedding.from_pretrained(Tensor(ifeatures), freeze=True)

        # embedding layers
        self.dense_embedding = TokenEmbedding(
            vocab_size=num_dense_item + 1,
            embedding_dim=hidden_dim
        )
        self.sparse_embedding = TokenEmbedding(
            vocab_size=num_items - num_dense_item + 1,
            embedding_dim=sparse_dim
        )

        # main layers
        self.upscaler = nn.Linear(sparse_dim, hidden_dim, bias=False)
        self.ac_encoder = nn.Linear(ifeature_dim + icontext_dim, hidden_dim * 4)
        self.item_encoder = nn.Linear(hidden_dim + hidden_dim * 4, hidden_dim)

    def forward(self,
                tokens: LongTensor,  # (b x L|C)
                icontexts: Tensor,  # (b x L|C x d_Ci)
                normalize: bool = True,
                ):

        # get ifeatures from cache
        ifeatures = self.ifeature_cache(tokens)

        # get ac vector
        ac = torch_cat([ifeatures, icontexts], dim=-1)
        ac_vector = self.ac_encoder(ac)

        # get token vector
        b, L = tokens.size()
        dense_tokens = tokens.clone()
        dense_tokens[dense_tokens > self.num_dense_item] = 0
        dense_token_vector = self.dense_embedding(dense_tokens)
        sparse_tokens = tokens.clone()
        sparse_tokens = sparse_tokens - self.num_dense_item
        sparse_tokens[sparse_tokens <= 0] = 0
        sparse_token_vector = self.sparse_embedding(sparse_tokens)
        sparse_token_vector = self.upscaler(sparse_token_vector)
        dense_token_vector[sparse_tokens != 0] = sparse_token_vector[sparse_tokens != 0]
        token_vector = dense_token_vector.view(b, L, -1)

        # get item vector
        vector = torch_cat([token_vector, ac_vector], dim=-1)
        vector = self.item_encoder(vector)

        # optionals
        if normalize:
            vector = F_normalize(vector, p=2, dim=-1)

        return vector
