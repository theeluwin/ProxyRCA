import numpy as np
import torch.nn as nn

from typing import Optional

from torch import (
    cat as torch_cat,
    Tensor,
    LongTensor,
)
from torch.nn.functional import softmax as F_softmax

from tools.utils import fix_random_seed

from ..layers import TokenEmbedding


__all__ = (
    'ProxyItemEncoder',
)


class ProxyItemEncoder(nn.Module):

    def __init__(self,
                 ifeatures: np.ndarray,
                 ifeature_dim: int,
                 icontext_dim: int,
                 hidden_dim: int = 64,
                 num_proxy_item: int = 128,
                 num_known_item: int = 1024,
                 dropout_prob: float = 0.1,
                 random_seed: Optional[int] = None,
                 ):
        super().__init__()

        # data params
        self.ifeature_dim = ifeature_dim
        self.icontext_dim = icontext_dim

        # main params
        self.hidden_dim = hidden_dim
        self.num_proxy_item = num_proxy_item
        self.num_known_item = num_known_item

        # optional params
        self.dropout_prob = dropout_prob
        self.random_seed = random_seed

        # set seed
        if random_seed is not None:
            fix_random_seed(random_seed)

        # 0: padding token
        # 1 ~ V: item tokens
        # V+1 ~: unknown token (convert to 0 after mask)
        self.vocab_size = num_known_item + 1

        # ifeature cache
        self.ifeature_cache = nn.Embedding.from_pretrained(Tensor(ifeatures), freeze=True)

        # embedding layers
        self.proxy_item_embedding = nn.Embedding(
            num_embeddings=num_proxy_item,
            embedding_dim=hidden_dim,
        )

        # proxy layers
        self.proxy_item_bias = TokenEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=num_proxy_item,
        )
        self.proxy_item_selector = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(ifeature_dim + icontext_dim, hidden_dim * 4),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim * 4, num_proxy_item),
        )

        # main layers
        self.ifeature_encoder = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(ifeature_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.ac_encoder = nn.Linear(hidden_dim + icontext_dim, hidden_dim)
        self.item_encoder = nn.Linear(hidden_dim + hidden_dim, hidden_dim)

        # initialize (very important trick)
        nn.init.zeros_(self.proxy_item_bias.weight)

    def forward(self,
                tokens: LongTensor,  # (b x L|C)
                icontexts: Tensor,  # (b x L|C x d_Ci)
                ):

        # get ifeatures from cache
        ifeatures = self.ifeature_cache(tokens)

        # get ac vector
        ac = torch_cat([self.ifeature_encoder(ifeatures), icontexts], dim=-1)
        ac_vector = self.ac_encoder(ac)

        # get proxy weight from ac
        ac = torch_cat([ifeatures, icontexts], dim=-1)
        proxy_weights = self.proxy_item_selector(ac)  # (b x L|C x d_x) -> (b x L|C x P)

        # get proxy bias from token (change unknown to padding - no bias)
        tokens[tokens >= self.vocab_size] = 0
        proxy_biases = self.proxy_item_bias(tokens)  # (b x L|C) -> (b x L|C x P)

        # get proxy logit
        proxy_logits = proxy_weights + proxy_biases

        # get proxy prob
        proxy_probs = F_softmax(proxy_logits, dim=-1)  # (b x L|C x P)

        # get proxy vector
        proxy_vector = proxy_probs @ self.proxy_item_embedding.weight  # (b x L|C x P) @ (P x d) -> (b x L|C x d)

        # get item vector
        vector = torch_cat([proxy_vector, ac_vector], dim=-1)
        vector = self.item_encoder(vector)

        return vector
