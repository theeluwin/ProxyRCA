import random

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from torch.utils.data import (
    Dataset,
    DataLoader,
)


__all__ = (
    'fix_random_seed',
)


def fix_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def init_train_dataloader(dataset: Dataset, config: dict):
    return DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['envs']['CPU_COUNT'],
        pin_memory=True,
        drop_last=True,
    )


def init_eval_dataloader(dataset: Dataset, config: dict):
    return DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['envs']['CPU_COUNT'],
        pin_memory=True,
        drop_last=False,
    )
