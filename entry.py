import os
import json
import argparse
import multiprocessing

import torch

from solvers import *  # noqa: F401,F403


# default
ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = 'runs'
default_config = {
    'envs': {
        'RUN_ROOT': os.path.join(ROOT, RUNS),
        'DATA_ROOT': os.path.join(ROOT, 'data'),
        'RAW_ROOT': os.path.join(ROOT, 'raw'),
        'CPU_COUNT': max(8, multiprocessing.cpu_count() // 4),
        'GPU_COUNT': torch.cuda.device_count(),
    },
    'solver': 'ProxyRCASolver',
    'dataset': 'fashion',
    'dataloader': {
        'sequence_len': 35,  # depends on dataset
        'train_num_negatives': 100,
        'valid_num_negatives': 100,
        'random_cut_prob': 1.0,
        'replace_user_prob': 0.0,
        'replace_item_prob': 0.01,
        'random_seed': None,
    },
    'model': {
        'hidden_dim': 256,
        'temporal_dim': 32,
        'num_proxy_item': 128,
        'num_known_item': 0,
        'num_layers': 1,
        'num_heads': 4,
        'dropout_prob': 0.1,
        'temperature': 0.1,
        'random_seed': None,
    },
    'train': {
        'epoch': 200,
        'every': 10,
        'patience': 80,
        'batch_size': 128,
        'optimizer': {
            'algorithm': 'adamw',
            'lr': 1e-4,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.1,
            'amsgrad': False,
        },
    },
    'metric': {
        'ks_valid': [10],
        'ks_test': [1, 5, 10, 20, 50, 100],
        'pivot': 'NDCG@10',
    },
    'memo': "",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help="run to execute")
    return parser.parse_args()


def update_dict_diff(base, diff):
    for key, value in diff.items():
        if isinstance(value, dict) and value:
            partial = update_dict_diff(base.get(key, {}), value)
            base[key] = partial
        else:
            base[key] = diff[key]
    return base


if __name__ == '__main__':

    # args
    args: argparse.Namespace = parse_args()

    # settle dirs
    run_root: str = os.path.join(ROOT, RUNS)
    run_dir: str = os.path.join(run_root, args.name)
    if not os.path.isdir(run_root):
        raise Exception(f"You need to create a `{RUNS}` directory.")
    if not os.path.isdir(run_dir):
        raise Exception("You need to create your run directory.")

    # check config file
    final_config_path: str = os.path.join(run_dir, 'config.json')
    if not os.path.isfile(final_config_path):
        raise Exception("You need to create a `config.json` in your run directory.")

    # get and update config
    config: dict = dict(default_config)
    partial_names = args.name.split('/')
    for i in range(1, len(partial_names) + 1):
        partial_config_path = os.path.join(run_root, '/'.join(partial_names[:i]), 'config.json')
        if os.path.isfile(partial_config_path):
            with open(partial_config_path, 'r') as fp:
                partial_config: dict = json.load(fp)
                update_dict_diff(config, partial_config)

    # settle config
    config['name'] = args.name
    config['run_dir'] = run_dir

    # lock config
    with open(os.path.join(run_dir, 'config-lock.json'), 'w') as fp:
        json.dump(config, fp, indent=4)

    # run
    solver_class = globals()[config['solver']]
    solver = solver_class(config)
    solver.solve()
