import torch
from torch.utils.data import DataLoader

from utils import AudioProcessor
from utils import register_cls


@register_cls('data_loader.AudioDataLoader')
class AudioDataLoader(DataLoader):
    def __init__(self, data_loader_config, dataset):

        self.dataset = dataset
        self.audio_preprocessing = data_loader_config["audio_preprocessing"]
        self.audio_processor = AudioProcessor()

        super().__init__(
            dataset=self.dataset,
            batch_size=data_loader_config["batch_size"],
            shuffle=data_loader_config["shuffle"],
            collate_fn=self.collate_fn,
            num_workers=data_loader_config["num_workers"])

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
