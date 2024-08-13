#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import json
import random

import librosa
import torch
import torch.nn.functional as F
from data_handling.sampler import BySequenceBatchSampler, BySequenceLengthSampler
from data_handling.text_transform import text_preprocess
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler


def _load_json_file(files, blacklist=None):
    json_data = []
    audio_id = 0
    if blacklist is not None:
        with open(blacklist, "r") as f:
            blacklist = json.load(f)
    for file in files:
        with open(file, "r") as f:
            json_obj = json.load(f)
            if json_obj["num_captions_per_audio"] == 1:
                for item in json_obj["data"]:
                    if "FreeSound" in file and blacklist is not None:
                        if item["id"] in blacklist["FreeSound"]:
                            continue
                    elif (
                        "AudioSet" in file or "AudioCaps" in file
                    ) and blacklist is not None:
                        if item["id"] in blacklist["AudioSet"]:
                            continue
                    temp_dict = {
                        "audio": item["audio"],
                        "caption": item["caption"],
                        "id": audio_id,
                        "duration": item["duration"],
                    }
                    json_data.append(temp_dict)
                    audio_id += 1
            else:
                for item in json_obj["data"]:
                    if "Clotho" in file and blacklist is not None:
                        if item["id"] in blacklist["FreeSound"]:
                            continue
                    for i in range(1, json_obj["num_captions_per_audio"] + 1):
                        temp_dict = {
                            "audio": item["audio"],
                            "caption": item[f"caption_{i}"],
                            "id": audio_id,
                            "duration": item["duration"],
                        }
                        json_data.append(temp_dict)
                    audio_id += 1
    return json_data


class AudioLanguagePretrainDataset(Dataset):
    def __init__(self, json_files, audio_config, blacklist=None):
        self.json_data = _load_json_file(json_files, blacklist)
        self.lengths = [item["duration"] for item in self.json_data]

        self.sr = audio_config["sr"]
        if audio_config["max_length"] != 0:
            self.max_length = audio_config["max_length"] * self.sr
        else:
            self.max_length = 0
        with open(
            "/scratch/shared/nfs2/oncescu/shared-datasets/ESC-50/audio_id_to_pos_neg_eg.json",
            "r",
        ) as f:
            self.positives_negatives = json.load(f)

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        item = self.json_data[index]
        wav_path = item["audio"]
        # duration = item["duration"]
        waveform, _ = librosa.load(wav_path, sr=self.sr, mono=True)

        if self.max_length != 0:
            # if audio length is longer than max_length, we randomly crop it to mac length
            if waveform.shape[-1] > self.max_length:
                max_start = waveform.shape[-1] - self.max_length
                start = random.randint(0, max_start)
                waveform = waveform[start : start + self.max_length]

        caption = text_preprocess(item["caption"])
        audio_id = item["id"]
        # original_audio_id = item["audio"].split("/")[-1].split(".")[0]
        audio_key = wav_path.split("/")[-1]
        dict_pos_negatives = self.positives_negatives[audio_key]
        positive_sentences = random.sample(dict_pos_negatives["positive"], 2)
        positive_sentences = [
            text_preprocess(positive_sentence)
            for positive_sentence in positive_sentences
        ]
        negative_sentences = random.sample(dict_pos_negatives["negative"], 2)
        negative_sentences = [
            text_preprocess(negative_sentence)
            for negative_sentence in negative_sentences
        ]
        return (
            torch.tensor(waveform),
            caption,
            audio_id,
            positive_sentences + negative_sentences,
        )
        # return duration, caption, audio_id


def collate_fn(batch):
    wav_list = []
    text_list = []
    audio_idx_list = []
    max_length = max([i[0].shape[-1] for i in batch])
    for waveform, text, audio_idx in batch:
        if waveform.shape[-1] < max_length:
            pad_length = max_length - waveform.shape[-1]
            waveform = F.pad(waveform, [0, pad_length], "constant", 0.0)
        wav_list.append(waveform)
        text_list.append(text)
        audio_idx_list.append(audio_idx)

    waveforms = torch.stack(wav_list, dim=0)
    audio_idx = torch.tensor(audio_idx_list).type(torch.long)
    return waveforms, text_list, audio_idx


def collate_fn_extra(batch):
    wav_list = []
    text_list = []
    audio_idx_list = []
    extra_sentences_list = []
    max_length = max([i[0].shape[-1] for i in batch])
    for waveform, text, audio_idx, extra_sentences in batch:
        if waveform.shape[-1] < max_length:
            pad_length = max_length - waveform.shape[-1]
            waveform = F.pad(waveform, [0, pad_length], "constant", 0.0)
        wav_list.append(waveform)
        text_list.append(text)
        audio_idx_list.append(audio_idx)
        extra_sentences_list.append(extra_sentences)

    waveforms = torch.stack(wav_list, dim=0)
    audio_idx = torch.tensor(audio_idx_list).type(torch.long)
    return waveforms, text_list, audio_idx, extra_sentences_list


def pretrain_dataloader(
    config,
    bucket: bool = True,
    bucket_boundaries: tuple = (5, 30, 6),
    is_distributed: bool = False,
    num_tasks: int = 0,
    global_rank: int = 0,
):
    dataset = AudioLanguagePretrainDataset(
        config["json_files"], config["audio_args"], config["blacklist"]
    )
    if "SynCaps" in config["json_files"][0]:
        print("using extra collate fn")
        collate_fn_fct = collate_fn_extra
    else:
        collate_fn_fct = collate_fn
    if bucket:
        sampler = BySequenceLengthSampler(
            lengths=dataset.lengths,
            bucket_boundaries=bucket_boundaries,
            batch_size=config["data_args"]["batch_size"],
            drop_last=True,
            seed=config["seed"],
        )
        return DataLoader(
            dataset=dataset,
            batch_sampler=BySequenceBatchSampler(
                sampler, batch_size=config["data_args"]["batch_size"], drop_last=False
            ),
            shuffle=False,
            num_workers=config["data_args"]["num_workers"],
            collate_fn=collate_fn_fct,
        )
    elif is_distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=config["data_args"]["batch_size"],
        num_workers=config["data_args"]["num_workers"],
        pin_memory=False,
        sampler=sampler,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_fct,
    )
