import hashlib
import librosa
import numpy as np
import os
import random
import re
from torch.utils.data import Dataset
from utils import Singleton, LABEL_SILENCE, LABEL_UNKNOWN, DatasetType


class GSCDatasetPreprocessor(metaclass=Singleton):

    def __init__(self, config):
        super(GSCDatasetPreprocessor, self).__init__()

        # organize audio files by class
        unknown_class_name = "_UNKNOWN_"

        audio_files_by_class = {}
        audio_counts_by_class = {}

        for class_name in os.listdir(config["data_dir"]):
            dir_path = os.path.join(config["data_dir"], class_name)

            if not os.path.isdir(dir_path):
                continue

            if class_name not in config["target_class"] and class_name != "_background_noise_":
                class_name = unknown_class_name

            if class_name not in audio_files_by_class:
                audio_files_by_class[class_name] = []
                audio_counts_by_class[class_name] = 0

            count = 0
            for file_name in os.listdir(dir_path):

                if "wav" not in file_name:
                    continue

                count += 1
                file_path = os.path.join(dir_path, file_name)
                audio_files_by_class[class_name].append(file_path)

            audio_counts_by_class[class_name] += count

        self.noise_files = audio_files_by_class.pop("_background_noise_")

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

            audio_files_by_class.pop(class_name)

        # unknwon class
        if config["unknown_class"]:
            unknwon_label = len(config["target_class"])
            for dataset in DatasetType:
                unknown_size = int(len(self.labels_by_dataset[dataset]) / len(self.label_mapping.keys()))
                self.audio_files_by_dataset[dataset] += random.sample(audio_files_by_class[unknown_class_name], unknown_size)
                self.labels_by_dataset[dataset] += ([unknwon_label] * unknown_size)
            self.label_mapping[unknwon_label] = LABEL_UNKNOWN

        # silence class
        if config["silence_class"]:
            silence_label = len(config["target_class"]) + 1
            for dataset in DatasetType:
                silence_size = int(len(self.labels_by_dataset[dataset]) / len(self.label_mapping.keys()))
                self.audio_files_by_dataset[dataset] += ([LABEL_SILENCE] * silence_size)
                self.labels_by_dataset[dataset] += ([silence_label] * silence_size)
            self.label_mapping[silence_label] = LABEL_SILENCE


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


class GSCDataset(Dataset):
    def __init__(self, config):
        super(GSCDataset, self).__init__()
        dataset = GSCDatasetPreprocessor(config)

        type = config["type"]
        self.sample_rate = config["sample_rate"]
        self.audio_files = dataset.audio_files_by_dataset[type]
        self.labels = dataset.labels_by_dataset[type]
        self.label_mapping = dataset.label_mapping

        self.noises = []

        for noise_file in dataset.noise_files:
            noise = librosa.core.load(noise_file, sr=self.sample_rate)[0]
            self.noises.append(noise * config["noise_pct"])

    def add_noise(self, audio):
        noise = random.choice(self.noises)
        start_pos = random.randint(0, len(noise) - self.sample_rate - 1)
        noise = noise[start_pos:start_pos + self.sample_rate]

        return audio + noise

    def __getitem__(self, index):
        label = self.labels[index]
        if LABEL_SILENCE == self.label_mapping[label]:
            data = np.zeros(self.sample_rate)
        else:
            file_path = self.audio_files[index]
            data = librosa.core.load(file_path, sr=self.sample_rate)[0]
            data = np.pad(data, (0, max(0, self.sample_rate - len(data))), "constant")

        data = self.add_noise(data)

        return data, label

    def __len__(self):
        return len(self.labels)
