import os
import torch
import numpy as np
import librosa
import random

from utils import AudioProcessor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class GSCDataset(Dataset):
    def __init__(self, config):
        super(GSCDataset, self).__init__()
        self.sample_rate = config['sample_rate']

        target_classes = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

        self.audio_files = []
        self.labels = []

        for class_name in os.listdir(config["data_dir"]):
            dir_path = os.path.join(config["data_dir"], class_name)
            
            if not os.path.isdir(dir_path) or class_name not in target_classes:
                continue

            count = 0
            for file_name in os.listdir(dir_path):

                if "wav" not in file_name:
                    continue

                if random.random() > config["dataset_ratio"]:
                    continue

                count += 1
                file_path = os.path.join(dir_path, file_name)
                self.audio_files.append(file_path)

            self.labels += [target_classes.index(class_name)] * count
            print(f"{class_name}: {count}")

        print(f"total audio counts: {len(self.audio_files)}")
        print(f"total label counts: {len(self.labels)}")

    def load_audio(self, file_path):
        data = librosa.core.load(file_path, sr=self.sample_rate)[0]
        data = np.pad(data, (0, max(0, self.sample_rate - len(data))), "constant")
        return data

    def __getitem__(self, index):
        return self.load_audio(self.audio_files[index]), self.labels[index]

    def __len__(self):
        return len(self.labels)

class GSCDataLoader(DataLoader):
    def __init__(self, config):
        self.dataset = GSCDataset(config)
        self.audio_preprocessing = config["audio_preprocessing"]
        self.audio_processor = AudioProcessor()

        super(GSCDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        processed = None
        targets = []
        for sample, label in batch:
            if self.audio_preprocessing == "MFCCs":
                audio_tensor = torch.from_numpy(self.audio_processor.compute_mfccs(sample).reshape(1, -1, 40))
                processed = audio_tensor if processed is None else torch.cat((processed, audio_tensor), 0)
            elif self.audio_preprocessing == "PCEN":
                audio_tensor = torch.from_numpy(np.expand_dims(sample, axis=0))
                audio_tensor = self.audio_processor.compute_pcen(audio_tensor)
                processed = audio_tensor if processed is None else torch.cat((processed, audio_tensor), 0)
            targets.append(label)
        return processed, torch.tensor(targets)
