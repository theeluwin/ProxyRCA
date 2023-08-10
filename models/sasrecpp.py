import numpy as np
import torch.nn as nn

from typing import Optional

from torch import (
    bmm as torch_bmm,
    tril as torch_tril,
    ones as torch_ones,
    bool as torch_bool,
    Tensor,
    LongTensor,
)
from torch.nn.functional import normalize as F_normalize

from tools.utils import fix_random_seed

from .layers import Transformer
from .encoders import AdvancedItemEncoder


__all__ = (
    'SASRecPP',
)


class SASRecPP(nn.Module):

    def __init__(self,
                 num_items: int,
                 ifeatures: np.ndarray,
                 ifeature_dim: int,
                 icontext_dim: int,
                 hidden_dim: int = 256,
                 num_known_item: Optional[int] = None,
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
        self.num_known_item = num_known_item
        self.num_layers = num_layers
        self.num_heads = num_heads

        # optional params
        self.dropout_prob = dropout_prob
        self.random_seed = random_seed

        # set seed
        if random_seed is not None:
            fix_random_seed(random_seed)

        # main layers
        self.item_encoder = AdvancedItemEncoder(
            num_items=num_items,
            ifeatures=ifeatures,
            ifeature_dim=ifeature_dim,
            icontext_dim=icontext_dim,
            hidden_dim=hidden_dim,
            num_known_item=num_known_item,
            random_seed=random_seed,
        )
        self.transformers = nn.ModuleList([
            Transformer(
                dim_model=hidden_dim,
                dim_ff=hidden_dim * 4,
                num_heads=num_heads,
                dropout_prob=dropout_prob
            ) for _ in range(num_layers)
        ])

    def forward(self,
                profile_tokens: LongTensor,  # (b x L)
                profile_icontexts: Tensor,  # (b x L x d_Ci)
                extract_tokens: LongTensor,  # (b x C)
                extract_icontexts: Tensor,  # (b x C x d_Ci)
                ):

        # length
        device = profile_tokens.device
        b = profile_tokens.size(0)
        L = profile_tokens.size(1)

        # mask for whether padding token or not in attention matrix (True if padding token)
        # dim: (b x L) -> (b x 1 x L) -> (b x L x L) -> (b x 1 x L x L)
        profile_token_mask = ~(profile_tokens > 0).unsqueeze(1).repeat(1, L, 1).unsqueeze(1)

        # autoregressive mask (True if ignore)
        # [process] (L x L) diagonal -> (1 x L x L) -> (b x L x L) -> (b x 1 x L x L)
        ar_mask = ~torch_tril(torch_ones((L, L), dtype=torch_bool)).unsqueeze(0).repeat(b, 1, 1).unsqueeze(1).to(device)
        mask = profile_token_mask | ar_mask

        # get profile vectors
        # dim: (b x L x d)
        P = self.item_encoder(
            profile_tokens,
            profile_icontexts,
        )
        P = F_normalize(P, p=2, dim=2)

        # get extract vectors
        # dim: (b x C x d)
        E = self.item_encoder(
            extract_tokens,
            extract_icontexts,
        )
        E = F_normalize(E, p=2, dim=2)

        # apply multi-layered transformers
        # [process] (b x L x d) -> ... -> (b x L x d)
        for transformer in self.transformers:
            P = transformer(P, mask)
        P = F_normalize(P, p=2, dim=2)

        # get scores
        # dim: (b x C x d) @ (b x d x 1) -> (b x C)
        P = P[:, -1, :].unsqueeze(2)
        logits = torch_bmm(E, P).squeeze(2)

        return logits
