#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


from typing import List, Optional, Tuple

import torch

# from pytorch_lightning import LightningDataModule
from data_handling.caption_dataset import AudioCaptionDataset
from data_handling.pretrain_dataset import collate_fn, collate_fn_extra
from tools.utils import worker_init_fn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler


class AudioCaptionDataModule:
    def __init__(
        self,
        config: dict,
        dataset: str,
    ):
        super(AudioCaptionDataModule, self).__init__()

        audio_config = config["audio_args"]

        self.train_set = AudioCaptionDataset(
            audio_config,
            dataset,
            split="train",
            filename=config["data_args"]["train"],
        )

        self.val_set = AudioCaptionDataset(
            audio_config,
            dataset,
            split="val",
            filename=config["data_args"]["val"],
        )

        self.test_set = AudioCaptionDataset(
            audio_config,
            dataset,
            split="test",
            filename=config["data_args"]["test"],
        )

        self.batch_size = config["data_args"]["batch_size"]
        self.num_workers = config["data_args"]["num_workers"]

    def _get_sampler(self, dataset, shuffle, is_distributed, num_tasks, global_rank):
        # do not return a sampler if is not in distributed mode
        # a default RandomSampler is used in this case
        if not is_distributed:
            return None

        return DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )

    def train_dataloader(self, is_distributed=False, num_tasks=0, global_rank=0):
        sampler = self._get_sampler(
            dataset=self.train_set,
            shuffle=True,
            is_distributed=is_distributed,
            num_tasks=num_tasks,
            global_rank=global_rank,
        )
        shuffle = sampler is None

        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn_extra,
            drop_last=True,
            # worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            sampler=None,
            shuffle=False,
            collate_fn=collate_fn_extra,
            drop_last=False,
            # worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            sampler=None,
            shuffle=False,
            collate_fn=collate_fn_extra,
            drop_last=False,
            # worker_init_fn=worker_init_fn,
        )
