import torch
from .gsc_dataset import GSCDataset
from utils import AudioProcessor
from torch.utils.data import DataLoader

class AudioDataLoader(DataLoader):
    def __init__(self, data_loader_config, dataset_config):
        self.dataset = GSCDataset(dataset_config)

        self.audio_preprocessing = data_loader_config["audio_preprocessing"]
        self.audio_processor = AudioProcessor()

        super(AudioDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=data_loader_config["batch_size"],
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
