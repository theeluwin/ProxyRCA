import numpy as np
import torch.nn as nn

from typing import Optional

from torch import (
    bmm as torch_bmm,
    Tensor,
    LongTensor,
)
from torch.nn.functional import normalize as F_normalize

from tools.utils import fix_random_seed

from .layers import TokenEmbedding
from .encoders import ProxyItemEncoder


__all__ = (
    'ProxyBPRPP',
)


class ProxyBPRPP(nn.Module):

    def __init__(self,
                 num_users: int,
                 num_items: int,
                 ifeatures: np.ndarray,
                 ifeature_dim: int,
                 icontext_dim: int,
                 hidden_dim: int = 256,
                 num_proxy_item: int = 128,
                 num_known_item: int = 0,
                 dropout_prob: float = 0.1,
                 random_seed: Optional[int] = None
                 ):
        """
            Note that item index starts from 1.
            Use 0 label (ignore index in CE) to avoid learning unmasked(context, known) items.
        """
        super().__init__()

        # data params
        self.num_users = num_users
        self.num_items = num_items
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

        # main layers
        self.user_embedding = TokenEmbedding(
            vocab_size=num_users + 1,
            embedding_dim=hidden_dim,
        )
        self.user_linear = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.item_encoder = ProxyItemEncoder(
            ifeatures=ifeatures,
            ifeature_dim=ifeature_dim,
            icontext_dim=icontext_dim,
            hidden_dim=hidden_dim,
            num_proxy_item=num_proxy_item,
            num_known_item=num_known_item,
            dropout_prob=dropout_prob,
            random_seed=random_seed,
        )
        self.item_linear = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self,
                uindex: LongTensor,  # (b,)
                extract_tokens: LongTensor,  # (b x C)
                extract_icontexts: Tensor,  # (b x C x d_Ci)
                ):

        # get profile vector
        # dim: (b x d)
        P = self.user_embedding(uindex)
        P = self.user_linear(P)
        P = P.unsqueeze(2)
        P = F_normalize(P, p=2, dim=1)

        # get extract vectors
        # dim: (b x C x d)
        E = self.item_encoder(
            extract_tokens,
            extract_icontexts,
        )
        E = self.item_linear(E)
        E = F_normalize(E, p=2, dim=2)

        # get scores
        # dim: (b x C x d) @ (b x d x 1) -> (b x C)
        logits = torch_bmm(E, P).squeeze(2)

        return logits
