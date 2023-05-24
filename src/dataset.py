import pandas as pd
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

class SERDataset(Dataset):
    def __init__(self, data_path, model_path="facebook/wav2vec2-base-960h"):
        self.data = pd.read_csv(data_path)
        self.target_sampling_rate = 16000
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.loc[index]

        speech_array, base_sampling_rate = torchaudio.load(sample['path'])
        resampler = torchaudio.transforms.Resample(base_sampling_rate, self.target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        vector = self.processor(speech)['input_values'][0]
        target = np.array(sample['label'])
        return speech, vector, target

if __name__ == '__main__':

    import os

    default_path = os.getcwd().split(os.path.sep + 'src')[0]
    dataset_path = os.path.join(default_path, 'datasets')

    traindatasets = SERDataset(data_path = f"{dataset_path}/train_split.csv")
    speech, vector, target = traindatasets[0]