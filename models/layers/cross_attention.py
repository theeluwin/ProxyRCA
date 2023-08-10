import torch.nn as nn

from typing import Optional

from torch import Tensor

from .transformer import (
    LayerNorm,
    MultiHeadedAttention,
    PositionWiseFeedForward,
)


__all__ = (
    'CrossAttention',
)


class CrossAttention(nn.Module):

    def __init__(self,
                 dim_model: int = 256,
                 dim_ff: int = 1024,
                 num_heads: int = 4,
                 dropout_prob: float = 0.1,
                 ):
        super().__init__()

        # params
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

        # layers
        self.layernorm_E = LayerNorm(dim_model)
        self.layernorm_P = LayerNorm(dim_model)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.attention = MultiHeadedAttention(
            num_heads=num_heads,
            dim_model=dim_model,
            dropout_prob=dropout_prob,
        )
        self.pwff = PositionWiseFeedForward(
            dim_model=dim_model,
            dim_ff=dim_ff,
            dropout_prob=dropout_prob,
        )

    def forward(self,
                E: Tensor,
                P: Tensor,
                mask: Optional[Tensor] = None,
                ):
        E = self.layernorm_E(E)  # E as query
        P = self.layernorm_P(P)  # P as key, value
        r = self.attention(E, P, P, mask=mask)
        r = self.dropout(r)
        Y = r * E
        Y = self.pwff(Y)
        return Y
