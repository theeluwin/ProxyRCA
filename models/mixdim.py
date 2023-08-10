import numpy as np
import torch.nn as nn

from typing import Optional

from torch import (
    Tensor,
    LongTensor,
)

from tools.utils import fix_random_seed

from .layers import (
    LayerNorm,
    Transformer,
    CrossAttention,
)
from .encoders import MixdimItemEncoder


__all__ = (
    'Mixdim',
)


class Mixdim(nn.Module):

    def __init__(self,
                 num_items: int,
                 ifeatures: np.ndarray,
                 ifeature_dim: int,
                 icontext_dim: int,
                 hidden_dim: int = 256,
                 sparse_dim: int = 128,
                 num_dense_item: int = 0,
                 num_layers: int = 1,
                 num_heads: int = 4,
                 dropout_prob: float = 0.1,
                 random_seed: Optional[int] = None
                 ):
        """
            Note that item index starts from 1.
            Use 0 label (ignore index in CE) to avoid learning unmasked(context, known) items.
        """
        super().__init__()

        # data params
        self.num_items = num_items
        self.ifeature_dim = ifeature_dim
        self.icontext_dim = icontext_dim

        # main params
        self.hidden_dim = hidden_dim
        self.sparse_dim = sparse_dim
        self.num_dense_item = num_dense_item
        self.num_layers = num_layers
        self.num_heads = num_heads

        # optional params
        self.dropout_prob = dropout_prob
        self.random_seed = random_seed

        # set seed
        if random_seed is not None:
            fix_random_seed(random_seed)

        # main layers
        self.item_encoder = MixdimItemEncoder(
            num_items=num_items,
            ifeatures=ifeatures,
            ifeature_dim=ifeature_dim,
            icontext_dim=icontext_dim,
            hidden_dim=hidden_dim,
            sparse_dim=sparse_dim,
            num_dense_item=num_dense_item,
            dropout_prob=dropout_prob,
            random_seed=random_seed,
        )
        self.item_layernorm = LayerNorm(hidden_dim)
        self.transformers = nn.ModuleList([
            Transformer(
                dim_model=hidden_dim,
                dim_ff=hidden_dim * 4,
                num_heads=num_heads,
                dropout_prob=dropout_prob
            ) for _ in range(num_layers)
        ])
        self.cross_attention = CrossAttention(
            dim_model=hidden_dim,
            dim_ff=hidden_dim * 4,
            num_heads=num_heads,
            dropout_prob=dropout_prob
        )
        self.scorize = nn.Linear(hidden_dim, 1)

    def forward(self,
                profile_tokens: LongTensor,  # (b x L)
                profile_icontexts: Tensor,  # (b x L x d_Ci)
                extract_tokens: LongTensor,  # (b x C)
                extract_icontexts: Tensor,  # (b x C x d_Ci)
                ):

        # length
        L = profile_tokens.size(1)
        C = extract_tokens.size(1)

        # mask for whether padding token or not in attention matrix (True if padding token)
        # dim: (b x L) -> (b x 1 x L) -> (b x L x L) -> (b x 1 x L x L)
        profile_token_mask = ~(profile_tokens > 0).unsqueeze(1).repeat(1, L, 1).unsqueeze(1)

        # mask for whether padding token or not in cross-attention matrix (True if padding token)
        # dim: (b x L) -> (b x 1 x L) -> (b x C x L) -> (b x 1 x C x L)
        extract_token_mask = ~(profile_tokens > 0).unsqueeze(1).repeat(1, C, 1).unsqueeze(1)

        # get profile vectors
        # dim: (b x L x d)
        P = self.item_encoder(
            profile_tokens,
            profile_icontexts,
        )
        P = self.item_layernorm(P)

        # get extract vectors
        # dim: (b x C x d)
        E = self.item_encoder(
            extract_tokens,
            extract_icontexts,
        )
        E = self.item_layernorm(E)

        # apply multi-layered transformers
        # [process] (b x L x d) -> ... -> (b x L x d)
        for transformer in self.transformers:
            P = transformer(P, profile_token_mask)

        # cross-attention
        # dim: (b x C x d)
        Y = self.cross_attention(E, P, extract_token_mask)

        # get scores
        # dim: (b x C x d) -> (b x C)
        logits = self.scorize(Y).squeeze(-1)

        return logits
