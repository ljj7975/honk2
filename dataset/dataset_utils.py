import random
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from torch.utils.data import Dataset


class DatasetType(Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"

def shuffle_in_groups(a, b):
    assert len(a) == len(b)
    zipped = list(zip(a, b))
    random.shuffle(zipped)
    return zip(*zipped)

class StreamingDataset(ABC, Dataset):
    def __init__(self, config):
        super().__init__()

        # create random stream by shuffling audio files
        self.audio_files, self.labels = shuffle_in_groups(self.audio_files, self.labels)

        # calculate total number of samples
        samples_per_ms = int(self.sample_rate / 1000)
        self.shift_size = config['shift_size_ms'] * samples_per_ms
        self.window_size = config['window_size_ms'] * samples_per_ms
        self.num_samples = int((config['total_num_sampels'] - self.window_size) / self.shift_size)
        print(f"with the window size of {self.window_size} and shfit size {self.shift_size}, {type} has {self.num_samples} samples")

        # to trigger the __reset_state at the start of every iteration
        self.last_index = self.num_samples

    @abstractmethod
    def __load_sample(self, index):
        pass

    def __reset_state(self):
        self.loaded_data = np.array([])
        self.loaded_labels = []
        self.audio_file_idx = 0
        self.label_counter = len(self.label_mapping) * [0]

    def __getitem__(self, index):
        if index < self.last_index:
            self.__reset_state()

        # update the loaded data if it is not sufficient
        while len(self.loaded_labels) < self.window_size:
            audio_data, label = self.__load_sample(self.audio_file_idx)

            self.loaded_data = np.concatenate((self.loaded_data, audio_data), axis=0)
            prev_loaded_labels_size = len(self.loaded_labels)
            self.loaded_labels += len(audio_data) * [label]

            assert len(self.loaded_data) == len(self.loaded_labels)

            # update label_counter if necessary
            if prev_loaded_labels_size < self.window_size:
                multiples = min(self.window_size, len(self.loaded_data)) - prev_loaded_labels_size
                self.label_counter[label] += multiples

            self.audio_file_idx += 1

        # construct return pair
        audio_window = self.loaded_data[:self.window_size]

        # label of the largest portion is used for the target label
        max_label_count = 0;
        target_label = None;
        for label, count in enumerate(self.label_counter):
            if count > max_label_count:
                target_label = label
                max_label_count = count

        # discard used data
        for i in range(0, self.shift_size):
            self.label_counter[self.loaded_labels[i]] -= 1

        for i in range(0, self.shift_size):
            label_idx = self.window_size + i

            if len(self.loaded_labels) > label_idx:
                self.label_counter[self.loaded_labels[label_idx]] += 1

        self.loaded_labels = self.loaded_labels[self.shift_size:]
        self.loaded_data = self.loaded_data[self.shift_size:]

        # keep track the current process
        self.last_index = index

        return audio_window, target_label

    def __len__(self):
        return self.num_samples
