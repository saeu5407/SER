import pandas as pd
import numpy as np
import librosa

import torch
import torchaudio
from torchaudio.transforms import MFCC
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import Wav2Vec2Processor

class SERDataset(Dataset):
    def __init__(self, data_path, model_path="facebook/wav2vec2-base-960h"):
        self.data = pd.read_csv(data_path)
        self.target_sampling_rate = 16000
        self.processor = Wav2Vec2Processor.from_pretrained(model_path, sampling_rate=self.target_sampling_rate, return_tensors="pt", padding=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.loc[index]
        speech_array, base_sampling_rate = torchaudio.load(sample['path'])
        resampler = torchaudio.transforms.Resample(base_sampling_rate, self.target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()

        # Wav2Vec2Processor 전처리
        vector = self.processor(speech, sampling_rate=self.target_sampling_rate)['input_values'][0]

        # Target
        target = np.array([sample['label']])

        # Prosodic Feature Extract
        prosodic = self._get_prosodic_features(speech)

        # MFCC Feature Extract
        mfcc = self._get_mfcc_features(speech)

        return vector, target, prosodic, mfcc

    def _get_prosodic_features(self, speech):

        def find_stat(input_vector):
            return np.array([np.mean(input_vector), np.std(input_vector), np.max(input_vector), np.min(input_vector)])

        # Prosodic Feature Extract
        # Extract Energy, Pitch, Duration
        '''
        frame_length, step은 제외, default 사용
        지금은 통계치를 추출하지만, 나중에는 벡터 값을 사용해보자
        '''
        energy = librosa.feature.rms(y=speech)
        pitch, _ = librosa.piptrack(y=speech, sr=self.target_sampling_rate)
        duration = librosa.onset.onset_detect(y=speech, sr=self.target_sampling_rate)

        return np.concatenate([find_stat(energy), find_stat(pitch), find_stat(duration)])

    def _get_mfcc_features(self, speech):
        # MFCC Feature Extract
        '''0.02(20ms) * 16000 = 320과 유사한 2의 제곱값 -> 256, 512'''
        frame_step = int(self.target_sampling_rate * 0.01)  # 10ms
        mfcc = librosa.feature.mfcc(y=speech, sr=self.target_sampling_rate, n_fft=512, hop_length=frame_step, n_mfcc=13)
        return mfcc

def collate_fn(batch):

    vector, target, prosodic, mfcc = zip(*batch)

    vector = pad_sequence([torch.tensor(x) for x in vector], batch_first=True)
    target = pad_sequence([torch.tensor(y) for y in target], batch_first=True)
    prosodic = pad_sequence([torch.tensor(z) for z in prosodic], batch_first=True)
    mfcc = pad_sequence([torch.tensor(i).permute(1,0) for i in mfcc], batch_first=True).permute(0,2,1)
    return vector, target, prosodic, mfcc

def collate_fn_test(batch):

    vector, prosodic, mfcc = zip(*batch)

    vector = pad_sequence([torch.tensor(x) for x in vector], batch_first=True)
    prosodic = pad_sequence([torch.tensor(z) for z in prosodic], batch_first=True)
    mfcc = pad_sequence([torch.tensor(i).permute(1,0) for i in mfcc], batch_first=True).permute(1,2,0)
    return vector, prosodic, mfcc


if __name__ == '__main__':

    import os

    default_path = os.getcwd().split(os.path.sep + 'src')[0]
    dataset_path = os.path.join(default_path, 'datasets')

    traindatasets = SERDataset(data_path = f"{dataset_path}/train_split.csv")
    vector, target, prosodic, mfcc = traindatasets[0]

    train_data = DataLoader(traindatasets,
                            batch_size=4,
                            shuffle=True,
                            collate_fn=collate_fn,
                            num_workers=1
                            )

    for vector, label, prosodic, mfcc in train_data:
        print(vector.shape)
        print(label.shape)
        print(prosodic.shape)
        print(mfcc.shape)
        break