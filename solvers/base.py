import os
import json
import logging

import torch

from typing import (
    List,
    Dict,
)
from datetime import datetime as dt

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import (
    Dataset,
    DataLoader,
)

from datasets import (
    BPRContrastiveTrainDataset,
    LWPContrastiveTrainDataset,
    EvalDataset,
)
from tools.utils import (
    init_eval_dataloader,
    init_train_dataloader,
)
from tools.metrics import (
    METRIC_NAMES,
    calc_batch_rec_metrics_per_k,
)


__all__ = (
    'BaseSolver',
    'BaseLWPContrastiveSolver',
)


class BaseSolver:

    def __init__(self, config: dict):
        self.config = config

        # basic
        self.init_path()
        self.init_logger()
        self.init_device()

        # data
        self.init_dataloader()

        # model
        self.init_model()  # depends on dataloader
        self.init_criterion()  # depends on dataloader
        self.init_optimizer()  # depends on model

    def init_path(self) -> None:
        C = self.config

        self.log_dir = os.path.join(C['run_dir'], 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger_path = os.path.join(self.log_dir, 'solver.log')
        self.writer_path = os.path.join(self.log_dir, 'scalars.json')

        self.pth_dir = os.path.join(C['run_dir'], 'pths')
        os.makedirs(self.pth_dir, exist_ok=True)
        self.model_path = os.path.join(self.pth_dir, 'model.pth')
        self.check_path = os.path.join(self.pth_dir, 'checkpoint.pth')

        self.data_dir = os.path.join(C['run_dir'], 'data')
        os.makedirs(self.data_dir, exist_ok=True)

    def init_logger(self) -> None:
        C = self.config

        # tx writer
        self.writer = SummaryWriter(self.log_dir)

        # logging logger
        self.logger = logging.getLogger(C['name'])
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self.logger_path)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def init_device(self) -> None:
        C = self.config
        if C['envs']['GPU_COUNT']:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_optimizer(self) -> None:
        CTO = self.config['train']['optimizer']
        if CTO['algorithm'] == 'sgd':
            self.optimizer = torch.optim.SGD(  # type: ignore
                self.model.parameters(),  # type: ignore
                lr=CTO['lr'],
                momentum=CTO['momentum'],
                weight_decay=CTO['weight_decay'],
            )
        elif CTO['algorithm'] == 'adam':
            self.optimizer = torch.optim.Adam(  # type: ignore
                self.model.parameters(),  # type: ignore
                lr=CTO['lr'],
                betas=(
                    CTO['beta1'],
                    CTO['beta2'],
                ),
                weight_decay=CTO['weight_decay'],
                amsgrad=CTO['amsgrad'],
            )
        elif CTO['algorithm'] == 'adamw':
            self.optimizer = torch.optim.AdamW(  # type: ignore
                self.model.parameters(),  # type: ignore
                lr=CTO['lr'],
                betas=(
                    CTO['beta1'],
                    CTO['beta2'],
                ),
                weight_decay=CTO['weight_decay'],
                amsgrad=CTO['amsgrad'],
            )
        else:
            raise NotImplementedError

    def load_model(self, purpose: str) -> None:
        C = self.config

        if purpose == 'test':
            self.logger.info(f"loading model with the best score: {C['name']}")
            model_dict = torch.load(self.model_path)  # if exception, just raise it
            self.model.load_state_dict(model_dict)  # type: ignore
            self.patience_counter = C['train']['patience'] + 1

        elif purpose == 'train':
            if os.path.isfile(self.check_path):
                self.logger.info(f"loading model from the checkpoint: {C['name']}")
                check = torch.load(self.check_path)  # if exception, just raise it
                self.start_epoch = check['epoch'] + 1
                self.best_score = check.get('best_score', None)
                self.patience_counter = check.get('patience_counter', 0)
                self.model.load_state_dict(check['model'])  # type: ignore
                self.optimizer.load_state_dict(check['optimizer'])
            else:
                self.logger.info(f"preparing new model from scratch: {C['name']}")
                self.start_epoch = 1
                self.best_score = None
                self.patience_counter = 0

    def set_model_mode(self, purpose: str) -> None:
        if purpose == 'eval':
            self.model = self.model.eval()  # type: ignore
        elif purpose == 'train':
            self.model = self.model.train()  # type: ignore

    def solve(self) -> None:
        C = self.config
        name = C['name']
        print(f"solve {name} start")

        # report new session
        self.logger.info("-- new solve session started --")
        dto_start = dt.now()

        # report model parameters
        numels = []
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                numels.append(parameter.numel())
        num_params = sum(numels)
        self.logger.info(f"total parameters:\t{num_params}")
        with open(os.path.join(self.data_dir, 'meta.json'), 'w') as fp:
            json.dump({'num_params': num_params}, fp)

        # backup
        self.backup()

        # sanity check
        self.load_model('train')
        if self.patience_counter >= C['train']['patience']:
            self.early_stop = True
        else:
            self.early_stop = False
        if self.start_epoch == 1:
            self.solve_valid(0)

        # solve loop
        if not self.early_stop:
            for epoch in range(self.start_epoch, C['train']['epoch'] + 1):
                self.solve_train(epoch)
                if not epoch % C['train']['every']:
                    self.solve_valid(epoch)
                if self.early_stop:
                    break

        # solve end
        self.load_model('test')
        self.solve_test()

        # verbose print (for excel, paper, etc.)
        dto_end = dt.now()
        self.summarize(dto_end - dto_start)

        # save writer
        self.writer.export_scalars_to_json(self.writer_path)
        self.writer.close()

        print(f"solve {name} end")

    def solve_train(self, epoch: int) -> None:
        self.train_one_epoch(epoch)

    def solve_valid(self, epoch: int) -> None:
        self.evaluate_valid(epoch)

    def solve_test(self) -> None:
        self.evaluate_test()

    def summarize(self, tdo) -> None:
        line = []
        message = ""
        with open(os.path.join(self.data_dir, 'meta.json')) as fp:
            meta = json.load(fp)
            line.append(meta['num_params'])
        with open(os.path.join(self.data_dir, 'results_mean.json')) as fp:
            results = json.load(fp)
            line.append(results['Recall@10'])
            line.append(results['NDCG@10'])
            message += f"N@10: {results['NDCG@10'] * 100:.02f}\n"
        line.append(str(tdo).split('.')[0])
        self.logger.info('\n' + '\t'.join([str(value) for value in line]))

    def train_one_epoch(self, epoch: int) -> None:
        self.logger.info(f"<train one epoch (epoch: {epoch})>")

        # prepare
        self.set_model_mode('train')
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        # loop
        pbar = tqdm(self.train_dataloader)
        pbar.set_description(f"[epoch {epoch} train]")
        for batch in pbar:

            # get loss
            loss = self.calculate_loss(batch)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # step end
            epoch_loss_sum += float(loss.data)
            epoch_loss_count += 1
            pbar.set_postfix(loss=f'{epoch_loss_sum / epoch_loss_count:.4f}')

        pbar.close()

        # epoch end
        epoch_loss = epoch_loss_sum / epoch_loss_count
        self.logger.info('\t'.join([
            f"[epoch {epoch}]",
            f"train loss: {epoch_loss:.4f}"
        ]))
        self.writer.add_scalar('train/loss', epoch_loss, epoch)

    def evaluate_valid(self, epoch: int) -> None:
        self.logger.info(f"<evaluate valid (epoch: {epoch})>")
        C = self.config

        # prepare
        self.set_model_mode('eval')
        self.max_top_k = max(C['metric']['ks_valid'])
        ks = sorted(C['metric']['ks_valid'])
        pivot = C['metric']['pivot']

        # init
        result_values = []

        # loop
        pbar = tqdm(self.valid_dataloader)
        pbar.set_description(f"[epoch {epoch} valid]")
        with torch.no_grad():
            for batch in pbar:

                # get rankers
                rankers = self.calculate_rankers(batch)

                # evaluate
                labels = batch['labels'].to(self.device)
                metric_values = calc_batch_rec_metrics_per_k(rankers, labels, ks)  # type: ignore
                result_values.extend(metric_values[pivot])

                # verbose
                pbar.set_postfix(pivot_score=f'{sum(result_values) / len(result_values):.4f}')

        pbar.close()

        # report
        pivot_score = sum(result_values) / len(result_values)
        self.logger.info('\t'.join([
            f"[epoch {epoch}]",
            f"valid pivot score: {pivot_score:.4f}"
        ]))
        self.writer.add_scalar('valid/pivot_score', pivot_score, epoch)

        # save best
        if self.best_score is None or self.best_score < pivot_score:
            self.best_score = pivot_score
            torch.save(self.model.state_dict(), self.model_path)
            self.logger.info(f"updated best model at epoch {epoch} with pivot score {pivot_score}")
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        if self.patience_counter >= C['train']['patience']:
            self.logger.info(f"no update for {self.patience_counter} epochs, thus early stopping")
            self.early_stop = True

        # save checkpoint
        torch.save({
            'epoch': epoch,
            'best_score': self.best_score,
            'patience_counter': self.patience_counter,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, self.check_path)

    def evaluate_test(self) -> None:
        self.logger.info("<evaluate test>")
        C = self.config

        # prepare
        self.set_model_mode('eval')
        self.max_top_k = max(C['metric']['ks_test'])
        ks = sorted(C['metric']['ks_test'])

        # init
        results_values: Dict[str, List] = {}
        metric_keys = []
        for k in ks:
            for metric_name in METRIC_NAMES:
                metric_key = f'{metric_name}@{k}'
                metric_keys.append(metric_key)
                results_values[metric_key] = []

        # loop
        pbar = tqdm(self.test_dataloader)
        pbar.set_description("[test]")
        with torch.no_grad():
            for batch in pbar:

                # get rankers
                rankers = self.calculate_rankers(batch)

                # evaluate
                labels = batch['labels'].to(self.device)
                batch_results = calc_batch_rec_metrics_per_k(rankers, labels, ks)  # type: ignore
                for metric_key in metric_keys:
                    results_values[metric_key].extend(batch_results[metric_key])

        pbar.close()

        # average
        results_mean: Dict[str, float] = {}
        for metric_key in metric_keys:
            values = results_values[metric_key]
            results_mean[metric_key] = sum(values) / len(values)

        # report
        reports = ["metric report (neg):"]
        reports.append('\t'.join(['k'] + list(METRIC_NAMES)))
        for k in ks:
            row = [str(k)]
            for metric_name in METRIC_NAMES:
                result = results_mean[f'{metric_name}@{k}']
                row.append(f"{result:.05f}",)
            reports.append('\t'.join(row))
        self.logger.info('\n'.join(reports))

        # save result
        with open(os.path.join(self.data_dir, 'results_mean.json'), 'w') as fpw:
            json.dump(results_mean, fpw)

    # override this
    def init_criterion(self) -> None:
        raise NotImplementedError

    # extend this
    def init_dataloader(self) -> None:
        C = self.config
        CD = C['dataloader']
        self.train_dataset: Dataset
        self.valid_dataset = EvalDataset(
            name=C['dataset'],
            target='valid',
            sequence_len=CD['sequence_len'],
            valid_num_negatives=CD['valid_num_negatives'],
            random_seed=CD['random_seed'],
        )
        self.test_dataset = EvalDataset(
            name=C['dataset'],
            target='test',
            sequence_len=CD['sequence_len'],
        )
        self.train_dataloader: DataLoader
        self.valid_dataloader = init_eval_dataloader(self.valid_dataset, C)
        self.test_dataloader = init_eval_dataloader(self.test_dataset, C)

    # override this
    def init_model(self) -> None:
        self.model: torch.nn.Module  # type: ignore
        raise NotImplementedError

    # override this
    def calculate_loss(self, batch):
        raise NotImplementedError

    # override this
    def calculate_rankers(self, batch):
        raise NotImplementedError

    # override this
    def backup(self):
        pass


class BaseLWPContrastiveSolver(BaseSolver):

    # override
    def init_criterion(self) -> None:
        self.ce_losser = CrossEntropyLoss()

    # extend
    def init_dataloader(self) -> None:
        super().init_dataloader()
        C = self.config
        CD = C['dataloader']
        self.train_dataset = LWPContrastiveTrainDataset(
            name=C['dataset'],
            sequence_len=CD['sequence_len'],
            random_cut_prob=CD['random_cut_prob'],
            replace_user_prob=CD['replace_user_prob'],
            replace_item_prob=CD['replace_item_prob'],
            train_num_negatives=CD['train_num_negatives'],
            random_seed=CD['random_seed']
        )
        self.train_dataloader = init_train_dataloader(self.train_dataset, C)


class BaseBPRContrastiveSolver(BaseSolver):

    # override
    def init_criterion(self) -> None:
        self.ce_losser = CrossEntropyLoss()

    # extend
    def init_dataloader(self) -> None:
        super().init_dataloader()
        C = self.config
        CD = C['dataloader']
        self.train_dataset = BPRContrastiveTrainDataset(
            name=C['dataset'],
            replace_user_prob=CD['replace_user_prob'],
            replace_item_prob=CD['replace_item_prob'],
            train_num_negatives=CD['train_num_negatives'],
            random_seed=CD['random_seed']
        )
        self.train_dataloader = init_train_dataloader(self.train_dataset, C)
