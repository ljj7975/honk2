import os
import random
from pathlib import Path

import librosa
import numpy as np
from torch.utils.data import Dataset

from utils import Singleton, DatasetType, load_json, register_cls


class HeySnipsDatasetPreprocessor(metaclass=Singleton):
    def __init__(self, config):
        super().__init__()

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
        self.label_mapping = {
            0: "Negative",
            1: "Positive"
        }

        # 9 seconds audio

        for dataset in DatasetType:
            json_file = Path(config["data_dir"], f"{dataset.value}.json")
            metadata = load_json(json_file)

            for data in metadata:
                # duration = float(data['duration'])
                label = data['is_hotword']

                file_name = "{}/{}".format(config["data_dir"], data["audio_file_path"])

                self.audio_files_by_dataset[dataset].append(file_name)
                self.labels_by_dataset[dataset].append(label)

                # for testing
                if len(self.audio_files_by_dataset[dataset]) > 10:
                    break;

@register_cls('dataset.heysnipsdataset')
class HeySnipsDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        dataset = HeySnipsDatasetPreprocessor(config)

        type = config["type"]
        self.sample_rate = config["sample_rate"]
        self.num_sample = config["audio_length"] * self.sample_rate
        self.audio_files = dataset.audio_files_by_dataset[type]
        self.labels = dataset.labels_by_dataset[type]
        self.label_mapping = dataset.label_mapping

        self.noises = np.random.uniform(0, config["noise_pct"], (config["num_noise_sample"], self.num_sample))

    def __getitem__(self, index):
        label = self.labels[index]
        file_path = self.audio_files[index]
        data = librosa.core.load(file_path, sr=self.sample_rate)[0]

        if len(data) > self.num_sample:
            # trim the audio with length greater than config["audio_length"]
            start_pos = np.random.choice(len(data) - self.num_sample)
            data = data[start_pos:start_pos + self.num_sample]
        elif len(data) < self.num_sample:
            # pad audio with zeros to match the size; randomly locate the data
            pad_size = max(0, self.num_sample - len(data))
            left_pad_size = np.random.choice(pad_size)
            data = np.pad(data, (left_pad_size, pad_size - left_pad_size), "constant")

        data += random.choice(self.noises)

        return data, label

    def __len__(self):
        return len(self.labels)
