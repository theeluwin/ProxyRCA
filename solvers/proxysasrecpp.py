import os
import pickle

from shutil import copyfile

from torch import topk as torch_topk

from models import ProxySASRecPP

from .base import BaseLWPContrastiveSolver


__all__ = (
    'ProxySASRecPPSolver',
)


class ProxySASRecPPSolver(BaseLWPContrastiveSolver):

    # override
    def init_model(self) -> None:
        C = self.config
        CM = C['model']

        # get num items
        with open(os.path.join(C['envs']['DATA_ROOT'], C['dataset'], 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
            self.num_items = len(self.iid2iindex)

        # get ifeatures
        with open(os.path.join(C['envs']['DATA_ROOT'], C['dataset'], 'ifeatures.pkl'), 'rb') as fp:
            ifeatures = pickle.load(fp)

        if type(CM['num_known_item']) is float:
            num_known_item = int(self.num_items * CM['num_known_item'])
        else:
            num_known_item = CM['num_known_item']

        # init model
        self.model = ProxySASRecPP(
            ifeatures=ifeatures,
            ifeature_dim=ifeatures.shape[1],
            icontext_dim=self.train_dataset.icontext_dim,  # type: ignore
            hidden_dim=CM['hidden_dim'],
            num_proxy_item=CM['num_proxy_item'],
            num_known_item=num_known_item,
            num_layers=CM['num_layers'],
            num_heads=CM['num_heads'],
            dropout_prob=CM['dropout_prob'],
            random_seed=CM['random_seed'],
        ).to(self.device)

    def calculate_forward(self, batch):

        # device
        profile_tokens = batch['profile_tokens'].to(self.device)  # b x L
        profile_icontexts = batch['profile_icontexts'].to(self.device)  # b x L x d_Ci
        extract_tokens = batch['extract_tokens'].to(self.device)  # b x C
        extract_icontexts = batch['extract_icontexts'].to(self.device)  # b x C x d_Ci

        # forward
        logits = self.model(
            profile_tokens,
            profile_icontexts,
            extract_tokens,
            extract_icontexts,
        )  # b x C

        return logits

    # override
    def calculate_loss(self, batch):

        # device
        label = batch['label'].to(self.device)  # b

        # forward
        logits = self.calculate_forward(batch)  # b x C
        logits = logits / self.config['model']['temperature']

        # loss
        loss = self.ce_losser(logits, label)
        return loss

    # override
    def calculate_rankers(self, batch):

        # forward
        logits = self.calculate_forward(batch)  # b x C

        # ranking
        _, rankers = torch_topk(logits, self.max_top_k, dim=1)

        return rankers

    # override
    def backup(self):
        copyfile('models/encoders/proxy.py', f'{self.data_dir}/encoder.py')
        copyfile('models/proxysasrecpp.py', f'{self.data_dir}/model.py')
        copyfile('solvers/proxysasrecpp.py', f'{self.data_dir}/solver.py')
