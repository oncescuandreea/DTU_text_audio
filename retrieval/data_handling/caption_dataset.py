#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import json
import random

import librosa
import torch
from data_handling.text_transform import text_preprocess
from torch.utils.data import Dataset


class AudioCaptionDataset(Dataset):
    def __init__(self, audio_config, dataset, split, filename=None):
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.sr = audio_config["sr"]

        if filename is not None:
            print(f"using {filename} for {split} instead of default json file")
            json_path = f"data/{dataset}/json_files/{filename}.json"
        else:
            json_path = f"data/{dataset}/json_files/{split}.json"

        if audio_config["max_length"] != 0:
            self.max_length = audio_config["max_length"] * self.sr
        else:
            self.max_length = 0

        with open(json_path, "r") as f:
            json_obj = json.load(f)
            self.num_captions_per_audio = json_obj["num_captions_per_audio"]
            json_obj = json_obj["data"]

        if self.num_captions_per_audio == 1:
            self.captions = [item["caption"] for item in json_obj]
            self.wav_paths = [item["audio"] for item in json_obj]
            if self.dataset == "Clotho" and split == "train":
                self.ids = [item["id"] for item in json_obj]
            print("got to num_captions 1")
        elif self.num_captions_per_audio == 5:
            print("got to num_captions 5")
            self.captions = [
                item["caption_{}".format(i)] for item in json_obj for i in range(1, 6)
            ]
            self.wav_paths = [item["audio"] for item in json_obj for _ in range(1, 6)]
        else:
            raise ValueError("Incorrect num_captions_per_audio.")

        if dataset in ["AudioCaps", "NewDataset"]:
            with open(
                f"data/{dataset}/json_files/audio_id_to_pos_neg_eg.json",
                "r",
            ) as f:
                self.positives_negatives = json.load(f)
        elif dataset == "Clotho":
            if self.split == "train":
                with open(
                    "data/Clotho/json_files/audio_id_to_pos_neg_eg_existent.json",
                    "r",
                ) as f:
                    self.positives_negatives = json.load(f)
            else:
                self.positives_negatives = None
        else:
            self.positives_negatives = None

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        audio_id = index // self.num_captions_per_audio
        audio_name = self.wav_paths[index].split("/")[-1]
        actual_audio_id = (
            self.ids[audio_id]
            if self.dataset == "Clotho" and self.split == "train"
            else None
        )
        wav_path = self.wav_paths[index]

        waveform, _ = librosa.load(wav_path, sr=self.sr, mono=True)

        if self.max_length != 0:
            # if audio length is longer than max_length, we random crop it
            if waveform.shape[-1] > self.max_length:
                # print(f'Got here for {audio_name}')
                max_start = waveform.shape[-1] - self.max_length
                start = random.randint(0, max_start)
                waveform = waveform[start : start + self.max_length]

        caption = text_preprocess(self.captions[index])
        if self.positives_negatives is not None:
            if self.dataset == "Clotho":
                dict_pos_negatives = self.positives_negatives[actual_audio_id]
            else:
                dict_pos_negatives = self.positives_negatives[audio_name]
            try:
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
            except ValueError:
                # sentence does not contain the future/past temporal cues
                positive_sentences = ["", ""]
                negative_sentences = ["", ""]
            # print(audio_name)
        if self.positives_negatives is not None:
            return (
                torch.tensor(waveform),
                caption,
                audio_id,
                positive_sentences + negative_sentences,
            )
        else:
            return torch.tensor(waveform), caption, audio_id, []
