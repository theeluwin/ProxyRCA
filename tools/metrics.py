from typing import (
    List,
    Dict,
)

from torch import (
    min as torch_min,
    arange as torch_arange,
    Tensor,
    LongTensor,
)


__all__ = (
    'METRIC_NAMES',
    'calc_batch_rec_metrics_per_k',
)


# METRIC_NAMES = ('HR', 'Recall', 'NDCG')
METRIC_NAMES = ('Recall', 'NDCG')


def calc_batch_rec_metrics_per_k(rankers: LongTensor,
                                 labels: LongTensor,
                                 ks: List[int]
                                 ) -> Dict[str, List[float]]:
    """
        Args:
            rankers: LongTensor, (b x M), index of each rank (0 to M-1)
            labels: LongTensor, (b x N), binary per each index (0 or 1)
            ks: list of top-k values

        Returns:
            a dict of various metrics.
            keys are 'HR@k', 'Recall@k', 'NDCG@k' for each ks.
            values are list of actual values; you should mean it by yourself, as needed.

        put'em all in the same device.
        no answer means zero.
    """

    # prepare
    metric_values: Dict[str, List[float]] = {}
    answer_count = labels.sum(1)
    device = labels.device
    ks = sorted(ks, reverse=True)

    # for each k
    for k in ks:

        rankers_at_k = rankers[:, :k]
        hit_per_rank = labels.gather(1, rankers_at_k)

        # # hr
        # hrs = hit_per_rank.sum(1).bool().float()
        # hrs_list = list(hrs.detach().cpu().numpy())
        # metric_values[f'HR@{k}'] = hrs_list

        # recall
        divisor = torch_min(
            Tensor([k]).to(device),
            answer_count,
        )
        recalls = (hit_per_rank.sum(1) / divisor.float())
        recalls[divisor == 0] = 0.0
        recalls_list = list(recalls.detach().cpu().numpy())
        metric_values[f'Recall@{k}'] = recalls_list

        # ndcg
        positions = torch_arange(1, k + 1).to(device).float()
        weights = 1 / (positions + 1).log2()
        dcg = (hit_per_rank * weights).sum(1)
        idcg = Tensor([weights[:min(n, k)].sum() for n in answer_count]).to(device)
        ndcgs = (dcg / idcg)
        ndcgs[idcg == 0] = 0.0
        ndcgs_list = list(ndcgs.detach().cpu().numpy())
        metric_values[f'NDCG@{k}'] = ndcgs_list

    return metric_values
