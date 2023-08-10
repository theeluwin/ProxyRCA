import os
import pickle
import random

import numpy as np

from typing import Optional

from torch import (
    Tensor,
    LongTensor,
)
from torch.utils.data import Dataset


__all__ = (
    'PlainTrainDataset',
    'BPRContrastiveTrainDataset',
    'LWPContrastiveTrainDataset',
    'EvalDataset',
    'ItemDataset',
)


class PlainTrainDataset(Dataset):

    data_root = 'data'

    def __init__(self, name: str):

        # params
        self.name = name

        # load data
        with open(os.path.join(self.data_root, name, 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
        with open(os.path.join(self.data_root, name, 'uindex2urows_train.pkl'), 'rb') as fp:
            self.uindex2urows_train = pickle.load(fp)

        # settle down
        self.uindices = list(self.uindex2urows_train.keys())
        self.num_items = len(self.iid2iindex)
        self.stamp_min = 9999999999
        self.stamp_max = 0
        for _, urows in self.uindex2urows_train.items():
            for _, stamp, _ in urows:
                if stamp > self.stamp_max:
                    self.stamp_max = stamp
                if stamp < self.stamp_min:
                    self.stamp_min = stamp
        self.stamp_interval = self.stamp_max - self.stamp_min

    def __len__(self):
        return len(self.uindices)

    def __getitem__(self, index):
        uindex = self.uindices[index]
        urows = self.uindex2urows_train[uindex]
        return {
            'uindex': uindex,
            'urows': urows,
        }

    @staticmethod
    def collate_fn(samples):
        uindex = [sample['uindex'] for sample in samples]
        urows = [sample['urows'] for sample in samples]
        return {
            'uindex': uindex,
            'urows': urows,
        }


class BPRContrastiveTrainDataset(Dataset):

    data_root = 'data'

    def __init__(self,
                 name: str,
                 replace_user_prob: float = 0.0,
                 replace_item_prob: float = 0.02,
                 train_num_negatives: int = 100,
                 random_seed: Optional[int] = None
                 ):

        # params
        self.name = name
        self.replace_user_prob = replace_user_prob
        self.replace_item_prob = replace_item_prob
        self.train_num_negatives = train_num_negatives
        self.random_seed = random_seed

        # rng
        if random_seed is None:
            self.rng = random.Random()
        else:
            self.rng = random.Random(random_seed)

        # load data
        with open(os.path.join(self.data_root, name, 'uid2uindex.pkl'), 'rb') as fp:
            self.uid2uindex = pickle.load(fp)
            self.uindex2uid = {uindex: uid for uid, uindex in self.uid2uindex.items()}
        with open(os.path.join(self.data_root, name, 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
            self.iindex2iid = {iindex: iid for iid, iindex in self.iid2iindex.items()}
        with open(os.path.join(self.data_root, name, 'uindex2urows_train.pkl'), 'rb') as fp:
            self.uindex2urows_train = pickle.load(fp)

        # settle down
        self.data = []
        self.uindices = []
        self.iindexset_train = set()
        self.uindex2iindexset = {}
        for uindex, urows in self.uindex2urows_train.items():
            iindexset_user = set()
            for iindex, stamp, icontext in urows:
                iindexset_user.add(iindex)
                self.iindexset_train.add(iindex)
                self.data.append((uindex, iindex, stamp, icontext))
            self.uindex2iindexset[uindex] = iindexset_user
            self.uindices.append(uindex)
        self.iindices_train = list(self.iindexset_train)
        self.num_items = len(self.iid2iindex)
        self.stamp_min = 9999999999
        self.stamp_max = 0
        for _, urows in self.uindex2urows_train.items():
            for _, stamp, _ in urows:
                if stamp > self.stamp_max:
                    self.stamp_max = stamp
                if stamp < self.stamp_min:
                    self.stamp_min = stamp
        self.stamp_interval = self.stamp_max - self.stamp_min

        # icontext info
        _, _, sample_icontext = urows[0]
        self.icontext_dim = len(sample_icontext)
        self.padding_icontext = np.zeros(self.icontext_dim)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # data point
        uindex, positive_iindex, positive_stamp, positive_icontext = self.data[index]
        iindexset_point = set(self.uindex2iindexset[uindex])
        num_iindices_train = len(self.iindices_train)

        # data driven regularization: replace user (see SSE-PT)
        if self.rng.random() < self.replace_user_prob:
            sampled_index = self.rng.randrange(0, len(self.uindices))
            uindex = self.uindices[sampled_index]

        # data driven regularization: replace item (see SSE)
        if self.rng.random() < self.replace_item_prob:
            sampled_index = self.rng.randrange(0, num_iindices_train)
            positive_iindex = self.iindices_train[sampled_index]
            iindexset_point.add(positive_iindex)

        # sample negatives
        extract_tokens = [positive_iindex]
        negative_tokens = set()
        while len(negative_tokens) < self.train_num_negatives:
            while True:
                sample_index = self.rng.randrange(0, num_iindices_train)
                negative_iindex = self.iindices_train[sample_index]
                if negative_iindex not in iindexset_point and negative_iindex not in negative_tokens:
                    break
            negative_tokens.add(negative_iindex)
        negative_tokens = list(negative_tokens)
        extract_tokens.extend(negative_tokens)

        # fill extract
        extract_stamps = []
        extract_icontexts = []
        for _ in extract_tokens:
            extract_stamps.append(positive_stamp)
            extract_icontexts.append(positive_icontext)

        # return tensorized data point
        return {
            'uindex': uindex,
            'extract_tokens': LongTensor(extract_tokens),
            'extract_stamps': LongTensor(extract_stamps),
            'extract_icontexts': Tensor(np.array(extract_icontexts)),
            'label': 0,
        }


class LWPContrastiveTrainDataset(Dataset):

    data_root = 'data'

    def __init__(self,
                 name: str,
                 sequence_len: int,
                 random_cut_prob: float = 1.0,
                 replace_user_prob: float = 0.0,
                 replace_item_prob: float = 0.02,
                 train_num_negatives: int = 100,
                 random_seed: Optional[int] = None
                 ):

        # params
        self.name = name
        self.sequence_len = sequence_len
        self.random_cut_prob = random_cut_prob
        self.replace_user_prob = replace_user_prob
        self.replace_item_prob = replace_item_prob
        self.train_num_negatives = train_num_negatives
        self.random_seed = random_seed

        # rng
        if random_seed is None:
            self.rng = random.Random()
        else:
            self.rng = random.Random(random_seed)

        # load data
        with open(os.path.join(self.data_root, name, 'uid2uindex.pkl'), 'rb') as fp:
            self.uid2uindex = pickle.load(fp)
            self.uindex2uid = {uindex: uid for uid, uindex in self.uid2uindex.items()}
        with open(os.path.join(self.data_root, name, 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
            self.iindex2iid = {iindex: iid for iid, iindex in self.iid2iindex.items()}
        with open(os.path.join(self.data_root, name, 'uindex2urows_train.pkl'), 'rb') as fp:
            self.uindex2urows_train = pickle.load(fp)

        # settle down
        self.uindices = []
        self.iindexset_train = set()
        self.uindex2iindexset = {}
        for uindex, urows in self.uindex2urows_train.items():
            iindexset_user = set()
            for iindex, _, _ in urows:
                iindexset_user.add(iindex)
                self.iindexset_train.add(iindex)
            self.uindex2iindexset[uindex] = iindexset_user
            if len(urows) < 2:
                continue
            self.uindices.append(uindex)
        self.iindices_train = list(self.iindexset_train)
        self.num_items = len(self.iid2iindex)
        self.stamp_min = 9999999999
        self.stamp_max = 0
        for _, urows in self.uindex2urows_train.items():
            for _, stamp, _ in urows:
                if stamp > self.stamp_max:
                    self.stamp_max = stamp
                if stamp < self.stamp_min:
                    self.stamp_min = stamp
        self.stamp_interval = self.stamp_max - self.stamp_min

        # tokens
        self.padding_token = 0

        # icontext info
        _, _, sample_icontext = urows[0]
        self.icontext_dim = len(sample_icontext)
        self.padding_icontext = np.zeros(self.icontext_dim)

    def __len__(self):
        return len(self.uindices)

    def __getitem__(self, index):

        # data point
        uindex = self.uindices[index]
        urows = self.uindex2urows_train[uindex]
        iindexset_point = set(self.uindex2iindexset[uindex])
        num_iindices_train = len(self.iindices_train)

        # data driven regularization: replace user (see SSE-PT)
        if self.rng.random() < self.replace_user_prob:
            sampled_index = self.rng.randrange(0, len(self.uindices))
            uindex = self.uindices[sampled_index]

        # long sequence random cut (see SSE-PT++)
        if self.rng.random() < self.random_cut_prob:
            urows = urows[:self.rng.randint(2, len(urows))]

        # last as positive
        positive_token, positive_stamp, positive_icontext = urows[-1]
        extract_tokens = [positive_token]

        # bake profile
        profile_tokens = []
        profile_stamps = []
        profile_icontexts = []
        for profile_iindex, profile_stamp, profile_icontext in urows[:-1][-self.sequence_len:]:

            # data driven regularization: replace item (see SSE)
            if self.rng.random() < self.replace_item_prob:
                sampled_index = self.rng.randrange(0, num_iindices_train)
                profile_iindex = self.iindices_train[sampled_index]
                iindexset_point.add(profile_iindex)

            # add item
            profile_tokens.append(profile_iindex)
            profile_stamps.append(profile_stamp)
            profile_icontexts.append(profile_icontext)

        # add paddings
        _, padding_stamp, _ = urows[0]
        padding_len = self.sequence_len - len(profile_tokens)
        profile_tokens = [self.padding_token] * padding_len + profile_tokens
        profile_stamps = [padding_stamp] * padding_len + profile_stamps
        profile_icontexts = [self.padding_icontext] * padding_len + profile_icontexts

        # sample negatives
        negative_tokens = set()
        while len(negative_tokens) < self.train_num_negatives:
            while True:
                sample_index = self.rng.randrange(0, num_iindices_train)
                negative_iindex = self.iindices_train[sample_index]
                if negative_iindex not in iindexset_point and negative_iindex not in negative_tokens:
                    break
            negative_tokens.add(negative_iindex)
        negative_tokens = list(negative_tokens)
        extract_tokens.extend(negative_tokens)

        # fill extract
        extract_stamps = []
        extract_icontexts = []
        for _ in extract_tokens:
            extract_stamps.append(positive_stamp)
            extract_icontexts.append(positive_icontext)

        # return tensorized data point
        return {
            'uindex': uindex,
            'profile_tokens': LongTensor(profile_tokens),
            'profile_stamps': LongTensor(profile_stamps),
            'profile_icontexts': Tensor(np.array(profile_icontexts)),
            'extract_tokens': LongTensor(extract_tokens),
            'extract_stamps': LongTensor(extract_stamps),
            'extract_icontexts': Tensor(np.array(extract_icontexts)),
            'label': 0,
        }


class EvalDataset(Dataset):

    data_root = 'data'

    def __init__(self,
                 name: str,
                 target: str,  # 'valid', 'test'
                 sequence_len: int,
                 valid_num_negatives: int = 100,
                 random_seed: Optional[int] = None
                 ):

        # params
        self.name = name
        self.target = target
        self.sequence_len = sequence_len
        self.valid_num_negatives = valid_num_negatives
        self.random_seed = random_seed

        # rng
        if random_seed is None:
            self.rng = random.Random()
        else:
            self.rng = random.Random(random_seed)

        # load data
        with open(os.path.join(self.data_root, name, 'uid2uindex.pkl'), 'rb') as fp:
            self.uid2uindex = pickle.load(fp)
            self.uindex2uid = {uindex: uid for uid, uindex in self.uid2uindex.items()}
        with open(os.path.join(self.data_root, name, 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
            self.iindex2iid = {iindex: iid for iid, iindex in self.iid2iindex.items()}
        with open(os.path.join(self.data_root, name, 'uindex2urows_train.pkl'), 'rb') as fp:
            self.uindex2urows_train = pickle.load(fp)
            self.iindexset_train = set()
            for uindex, urows in self.uindex2urows_train.items():
                for iindex, _, _ in urows:
                    self.iindexset_train.add(iindex)
        with open(os.path.join(self.data_root, name, 'uindex2urows_valid.pkl'), 'rb') as fp:
            self.uindex2urows_valid = pickle.load(fp)
            self.iindexset_valid = set()
            for uindex, urows in self.uindex2urows_valid.items():
                for iindex, _, _ in urows:
                    self.iindexset_valid.add(iindex)
        with open(os.path.join(self.data_root, name, 'uindex2urows_test.pkl'), 'rb') as fp:
            self.uindex2urows_test = pickle.load(fp)
            self.uindex2aiindexset_test = {}
            for uindex, urows in self.uindex2urows_test.items():
                aiindexset = set()
                for iindex, _, _ in urows:
                    aiindexset.add(iindex)
                self.uindex2aiindexset_test[uindex] = aiindexset
        with open(os.path.join(self.data_root, name, 'ns_random.pkl'), 'rb') as fp:
            self.uindex2negatives_test = pickle.load(fp)

        # settle down
        if target == 'valid':
            self.uindices = []
            for uindex in self.uindex2urows_valid:
                if uindex in self.uindex2urows_train:
                    self.uindices.append(uindex)
        elif target == 'test':
            self.uindices = []
            for uindex in self.uindex2aiindexset_test:
                if uindex not in self.uindex2urows_train and uindex not in self.uindex2urows_valid:
                    continue
                self.uindices.append(uindex)
        self.iindexset_known = set()
        self.uindex2iindexset = {}
        for uindex, urows in self.uindex2urows_train.items():
            iindexset_user = set()
            for iindex, _, _ in urows:
                iindexset_user.add(iindex)
                self.iindexset_known.add(iindex)
            self.uindex2iindexset[uindex] = iindexset_user
        for uindex, urows in self.uindex2urows_valid.items():
            iindexset_user = set()
            for iindex, _, _ in urows:
                iindexset_user.add(iindex)
                self.iindexset_known.add(iindex)
            if uindex not in self.uindex2iindexset:
                self.uindex2iindexset[uindex] = set()
            self.uindex2iindexset[uindex] |= iindexset_user
        self.iindices_known = list(self.iindexset_known)
        self.num_items = len(self.iid2iindex)

        # tokens
        self.padding_token = 0

        # icontext info
        _, _, sample_icontext = urows[0]
        self.icontext_dim = len(sample_icontext)
        self.padding_icontext = np.zeros(self.icontext_dim)

    def __len__(self):
        return len(self.uindices)

    def __getitem__(self, index):

        # get data point
        uindex = self.uindices[index]
        urows_train = self.uindex2urows_train.get(uindex, [])
        urows_valid = self.uindex2urows_valid.get(uindex, [])
        urows_test = self.uindex2urows_test.get(uindex, [])

        # prepare rows
        if self.target == 'valid':
            urows_known = urows_train
            urows_eval = urows_valid
        elif self.target == 'test':
            urows_known = urows_train + urows_valid
            urows_eval = urows_test

        # get eval row
        answer_iindex, answer_stamp, answer_icontext = urows_eval[0]
        extract_tokens = [answer_iindex]

        # bake profile
        profile_tokens = []
        profile_stamps = []
        profile_icontexts = []
        for profile_iindex, profile_stamp, profile_icontext in urows_known[-self.sequence_len:]:
            profile_tokens.append(profile_iindex)
            profile_stamps.append(profile_stamp)
            profile_icontexts.append(profile_icontext)

        # add paddings
        _, padding_stamp, _ = urows_known[0]
        padding_len = self.sequence_len - len(profile_tokens)
        profile_tokens = [self.padding_token] * padding_len + profile_tokens
        profile_stamps = [padding_stamp] * padding_len + profile_stamps
        profile_icontexts = [self.padding_icontext] * padding_len + profile_icontexts

        # sample negatives
        if self.target == 'valid':
            negative_tokens = set()
            iindexset_user = self.uindex2iindexset[uindex]
            num_iindices_known = len(self.iindices_known)
            while len(negative_tokens) < self.valid_num_negatives:
                while True:
                    sampled_index = self.rng.randrange(0, num_iindices_known)
                    negative_iindex = self.iindices_known[sampled_index]
                    if negative_iindex not in iindexset_user and negative_iindex not in negative_tokens:
                        break
                negative_tokens.add(negative_iindex)
            negative_tokens = list(negative_tokens)
        elif self.target == 'test':
            negative_tokens = self.uindex2negatives_test[uindex]
        extract_tokens.extend(negative_tokens)

        # bake extract
        extract_stamps = []
        extract_icontexts = []
        for _ in extract_tokens:
            extract_stamps.append(answer_stamp)
            extract_icontexts.append(answer_icontext)
        labels = [1] + [0] * (len(extract_tokens) - 1)

        # return tensorized data point
        return {
            'uindex': uindex,
            'profile_tokens': LongTensor(profile_tokens),
            'profile_stamps': Tensor(np.array(profile_stamps)),
            'profile_icontexts': Tensor(np.array(profile_icontexts)),
            'extract_tokens': LongTensor(extract_tokens),
            'extract_stamps': Tensor(np.array(extract_stamps)),
            'extract_icontexts': Tensor(np.array(extract_icontexts)),
            'labels': LongTensor(labels),
        }


class ItemDataset(Dataset):

    data_root = 'data'

    def __init__(self,
                 name: str,
                 ):

        # params
        self.name = name

        # load data
        with open(os.path.join(self.data_root, name, 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
            self.iindex2iid = {iindex: iid for iid, iindex in self.iid2iindex.items()}

        # settle down
        self.iindices = sorted(list(self.iid2iindex.values()))

    def __len__(self):
        return len(self.iindices)

    def __getitem__(self, index):
        return {
            'iindex': self.iindices[index],
        }
