import hashlib
import os
import random
import re
from pathlib import Path

import librosa
import numpy as np
from torch.utils.data import Dataset

from .dataset_utils import DatasetType
from utils import Singleton, register_cls


LABEL_SILENCE = "__silence__"
LABEL_UNKNOWN = "__unknown__"

class GSCDatasetPreprocessor(metaclass=Singleton):
    def __init__(self, config):
        super().__init__()

        # organize audio files by class
        unknown_class_name = "_UNKNOWN_"

        audio_files_by_class = {}
        audio_counts_by_class = {}

        for class_path in Path(config["data_dir"]).iterdir():

            if not class_path.is_dir():
                continue

            class_name = class_path.name

            if class_name not in config["target_class"] and class_name != "_background_noise_":
                class_name = unknown_class_name

            if class_name not in audio_files_by_class:
                audio_files_by_class[class_name] = []
                audio_counts_by_class[class_name] = 0

            count = 0
            for file_path in class_path.iterdir():

                if ".wav" != file_path.suffix:
                    continue

                count += 1
                audio_files_by_class[class_name].append(file_path.as_posix())

            audio_counts_by_class[class_name] += count

        noise_files = audio_files_by_class.pop("_background_noise_")

        # split the dataset into trian/dev/test
        self.bucket_size = 2**27 - 1
        self.dev_pct = config["dev_pct"]
        self.test_pct = config["test_pct"]

        self.audio_files_by_dataset = {
            DatasetType.TRAIN: [],
            DatasetType.DEV: [],
            DatasetType.TEST: []
        }
        self.labels_by_dataset = {
            DatasetType.TRAIN: [],
            DatasetType.DEV: [],
            DatasetType.TEST: []
        }
        self.label_mapping = {}

        # target class
        for class_name in config["target_class"]:
            audio_list = audio_files_by_class[class_name]

            label = config["target_class"].index(class_name)
            self.label_mapping[label] = class_name

            for audio_file in audio_list:
                bucket = self.get_bucket_from_file_name(audio_file, config["group_speakers_by_id"])
                self.distribute_to_dataset(bucket, audio_file, label)

        # unknown class
        if config["unknown_class"]:
            unknown_label = len(config["target_class"])
            for dataset in DatasetType:
                unknown_size = int(len(self.labels_by_dataset[dataset]) / len(self.label_mapping.keys()))
                self.audio_files_by_dataset[dataset] += random.sample(audio_files_by_class[unknown_class_name], unknown_size)
                self.labels_by_dataset[dataset] += ([unknown_label] * unknown_size)
            self.label_mapping[unknown_label] = LABEL_UNKNOWN

        # silence class
        if config["silence_class"]:
            silence_label = len(config["target_class"]) + 1
            for dataset in DatasetType:
                silence_size = int(len(self.labels_by_dataset[dataset]) / len(self.label_mapping.keys()))
                self.audio_files_by_dataset[dataset] += ([LABEL_SILENCE] * silence_size)
                self.labels_by_dataset[dataset] += ([silence_label] * silence_size)
            self.label_mapping[silence_label] = LABEL_SILENCE

        # noise samples
        self.noise_samples_by_dataset = {
            DatasetType.TRAIN: [],
            DatasetType.DEV: [],
            DatasetType.TEST: []
        }

        sample_rate = config["sample_rate"]
        for file_name in noise_files:
            full_noise = librosa.core.load(file_name, sr=sample_rate)[0]
            for i in range(0, len(full_noise)-sample_rate, sample_rate):
                noise_sample = full_noise[i:i + sample_rate] * random.random()

                bucket = random.random()
                if bucket < self.test_pct:
                    self.noise_samples_by_dataset[DatasetType.TEST].append(noise_sample)
                elif bucket < self.dev_pct + self.test_pct:
                    self.noise_samples_by_dataset[DatasetType.DEV].append(noise_sample)
                else:
                    self.noise_samples_by_dataset[DatasetType.TRAIN].append(noise_sample)


    def get_bucket_from_file_name(self, audio_file, group_speakers_by_id):
        if group_speakers_by_id:
            hashname_search = re.search(r"(\w+)_nohash_.*$", audio_file, re.IGNORECASE)
            if hashname_search:
                hashname = hashname_search.group(1)

            sha = int(hashlib.sha1(hashname.encode()).hexdigest(), 16)
            bucket = (sha % (self.bucket_size + 1)) / self.bucket_size
        else:
            bucket = random.random()

        return bucket

    def distribute_to_dataset(self, bucket, audio_file, label):
        if bucket < self.test_pct:
            self.audio_files_by_dataset[DatasetType.TEST].append(audio_file)
            self.labels_by_dataset[DatasetType.TEST].append(label)
        elif bucket < self.dev_pct + self.test_pct:
            self.audio_files_by_dataset[DatasetType.DEV].append(audio_file)
            self.labels_by_dataset[DatasetType.DEV].append(label)
        else:
            self.audio_files_by_dataset[DatasetType.TRAIN].append(audio_file)
            self.labels_by_dataset[DatasetType.TRAIN].append(label)

@register_cls('dataset.GSCDataset')
class GSCDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        dataset = GSCDatasetPreprocessor(config)

        type = config["type"]
        self.sample_rate = config["sample_rate"]
        self.audio_files = dataset.audio_files_by_dataset[type]
        self.labels = dataset.labels_by_dataset[type]
        self.label_mapping = dataset.label_mapping

        self.noises = dataset.noise_samples_by_dataset[type]
        self.noise_pct = config["noise_pct"]

    def __getitem__(self, index):
        label = self.labels[index]
        if LABEL_SILENCE == self.label_mapping[label]:
            data = np.zeros(self.sample_rate)
        else:
            file_path = self.audio_files[index]
            data = librosa.core.load(file_path, sr=self.sample_rate)[0]
            data = np.pad(data, (0, max(0, self.sample_rate - len(data))), "constant")

        data += (random.choice(self.noises) * self.noise_pct)

        return data, label

    def __len__(self):
        return len(self.labels)
