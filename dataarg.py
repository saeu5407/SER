import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns

def noise(data):
    """
    Adding White Noise
    ------------------
    White Noises are random sample distribution at
    regular intervals with mean of standard deviation of 1.

    """
    noise_amp = 0.05 * np.random.uniform() * np.random.normal(size=data.shape[0])
    # more noise reduce the value to 0.5
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data

def shift(data):
    """
    Random shifting
    ----------------
    The pixels of the image can be shifted horizontally or vertically.
    Here Image is mfcc spectrum
    """
    s_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, s_range)

def stretch(data, rate=0.8):
    """
    Streching the sound. Note that this expands the dataset slightly
    """
    data = librosa.effects.time_stretch(data, rate)
    return data

def pitch(data, sample_rate):
    """
    Pitch Tuning
    """
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    data = librosa.effects.pitch_shift(data.astype('float64'),
                                       sample_rate,
                                       n_steps=pitch_change,
                                       bins_per_octave=bins_per_octave
                                       )
    return data

def dyn_change(data):
    """
    Random value change
    """
    dyn_change = np.random.uniform(low=-0.5, high=7)
    return data * dyn_change

def speedNpitch(data):
    """
    speed and Pitch Tuning
    """
    length_change = np.random.uniform(low=0.8, high=1)
    speed_fac = 1.2 / length_change  # try changing 1.0 to 2.0...
    tmp = np.interp(np.arange(0, len(data), speed_fac), np.arange(0, len(data)), data)
    minlen = min(data.shape[0], tmp.shape[0])
    data *= 0
    data[0:minlen] = tmp[0:minlen]
    return data

# 데이터 로드
data = pd.read_csv('./train.csv')
data = data.loc[data.id != 'TRAIN_2143', :]
data = data.loc[:500,:] # TESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTEST
data.reset_index(drop=True, inplace=True)

train_data, valid_data = \
    train_test_split(data, test_size=0.2, stratify=data['label'], random_state=10)

n_fft = 1024
win_length = 1024
hop_length = 128
n_mels = 128

def make_train(data):

    n_fft = 1024
    win_length = 1024
    hop_length = 128
    n_mels = 128
    max_len = 672

    data_mfcc = []
    data_mel = []
    data_label = []
    mfcc_max = []
    for idx in tqdm(range(0, len(data))):
        path = data.path[idx]
        label = data.label[idx]

        waveform, sr = librosa.load(path)

        stft = np.abs(librosa.stft(waveform, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
        mel = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mels)
        mfcc = (mfcc - mfcc.mean()) / mfcc.std()
        data_mfcc.append(torch.tensor(mfcc))
        data_mel.append(torch.tensor(mel))
        data_label.append(label)
        mfcc_max.append(mfcc.shape[1])

        waveform = noise(waveform)
        stft = np.abs(librosa.stft(waveform, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
        mel = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mels)
        mfcc = (mfcc - mfcc.mean()) / mfcc.std()
        data_mfcc.append(torch.tensor(mfcc))
        data_mel.append(torch.tensor(mel))
        data_label.append(label)
        mfcc_max.append(mfcc.shape[1])

        waveform = speedNpitch(waveform)
        stft = np.abs(librosa.stft(waveform, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
        mel = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mels)
        mfcc = (mfcc - mfcc.mean()) / mfcc.std()
        data_mfcc.append(torch.tensor(mfcc))
        data_mel.append(torch.tensor(mel))
        data_label.append(label)
        mfcc_max.append(mfcc.shape[1])

    max_len = max(mfcc_max)

    tensor_mfcc = []
    for idx in data_mfcc:
        # Padding
        audio_length = idx.shape[1]
        padding_length = max_len - audio_length
        padded_mfcc = torch.nn.functional.pad(idx, (0, padding_length), value=0)
        tensor_mfcc.append(padded_mfcc)
    tensor_mfcc = torch.stack(tensor_mfcc)
    tensor_mfcc = tensor_mfcc.float()

    tensor_mel = []
    for idx in data_mel:
        # Padding
        audio_length = idx.shape[1]
        padding_length = max_len - audio_length
        padded_mel = torch.nn.functional.pad(idx, (0, padding_length), value=0)
        tensor_mel.append(padded_mel)
    tensor_mel = torch.stack(tensor_mel)
    tensor_mel = tensor_mel.float()

    tensor_label = torch.tensor(data_label)
    tensor_label = tensor_label.long()

    return tensor_mfcc, tensor_mel, tensor_label, max_len

train_audio, train_mel, train_label, max_len = make_train(train_data.reset_index(drop=True))

def make_valid(data, max_len):

    n_fft = 1024
    win_length = 1024
    hop_length = 128
    n_mels = 128

    tensor_mfcc = []
    tensor_mel = []
    data_label = []
    for idx in tqdm(range(0, len(data))):
        path = data.path[idx]
        label = data.label[idx]

        waveform, sr = librosa.load(path)

        stft = np.abs(librosa.stft(waveform, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
        mel = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
        mel = torch.tensor(mel)
        audio_length = mel.shape[1]
        padding_length = max_len - audio_length
        padded_mel = torch.nn.functional.pad(mel, (0, padding_length), value=0)
        tensor_mel.append(padded_mel)

        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mels)
        mfcc = (mfcc - mfcc.mean()) / mfcc.std()
        mfcc = torch.tensor(mfcc)
        audio_length = mfcc.shape[1]
        padding_length = max_len - audio_length
        padded_mfcc = torch.nn.functional.pad(mfcc, (0, padding_length), value=0)
        tensor_mfcc.append(padded_mfcc)

        data_label.append(label)

    tensor_mel = torch.stack(tensor_mel)
    tensor_mel = tensor_mel.float()

    tensor_mfcc = torch.stack(tensor_mfcc)
    tensor_mfcc = tensor_mfcc.float()

    tensor_label = torch.tensor(data_label)
    tensor_label = tensor_label.long()

    return tensor_mfcc, tensor_mel, tensor_label

valid_audio, valid_mel, valid_label = make_valid(valid_data.reset_index(drop=True), max_len)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy

class EmotionDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y

class EmotionClassifierWithCNN(nn.Module):
    def __init__(self):
        super(EmotionClassifierWithCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 6)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.gmp(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

batch_size = 32

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_name='test_model'):

    history = []
    best_valid_loss = np.NaN
    #pbar = tqdm(range(1, num_epochs + 1))
    #logger = logging.getLogger(__name__)
    #logger.setLevel(logging.INFO)
    #logger.addHandler(TqdmLoggingHandler(pbar=pbar))
    model.to(device)
    for epoch in range(1, num_epochs + 1):

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        history.append([epoch, train_loss, val_loss, train_accuracy, val_accuracy])


        #logger.info(f"Epoch : {epoch}, Loss : {train_loss:.4f}, Valid Loss : {val_loss:.4f}, Valid Acc : {val_accuracy:.2f}")
        print(f"Epoch : {epoch}, Loss : {train_loss:.4f}, Valid Loss : {val_loss:.4f}, Train Acc : {train_accuracy:.2f}, Valid Acc : {val_accuracy:.2f}")

        if (val_loss < best_valid_loss) or (epoch == 1):
            best_acc = val_accuracy
            best_epoch = epoch
            best_train_loss = train_loss
            best_valid_loss = val_loss
            best_train_acc = train_accuracy
            best_model = copy.deepcopy(model.state_dict())

    checkpoint = {
        'model_state_dict': best_model,
        'criterion_state_dict': criterion.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': best_epoch,
        'train_loss': float(best_train_loss),
        'valid_loss': float(best_valid_loss),
        'train_acc': float(best_train_acc),
        'valid_acc': float(best_acc)
    }
    torch.save(checkpoint, f'./models/{model_name}_{epoch}.pth')

    history = pd.DataFrame(history, columns=['epoch', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    history.to_csv(f'./models/{model_name}_history.csv', index=False)

    return best_model, history

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> Use {device}")

train_dataset = EmotionDataset(train_audio, train_label)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = EmotionDataset(valid_audio, valid_label)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

class EmotionClassifierWithCNN(nn.Module):
    def __init__(self):
        super(EmotionClassifierWithCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 6)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.gmp(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 모델 초기화
input_size = train_dataset[0][0].shape[-1]
hidden_size = 256
num_classes = 6
cnn_model = EmotionClassifierWithCNN()

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-4)
cnn_model, history = train(cnn_model, train_dataloader, valid_dataloader, criterion, optimizer, device, 100, 'cnn_model')

class EmotionClassifierWithLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EmotionClassifierWithLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        # self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        # out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])  # 마지막 시점의 출력만 사용
        return out

# 모델 초기화
input_size = train_dataset[0][0].shape[-1]
hidden_size = 256
num_classes = 6
lstm_model = EmotionClassifierWithLSTM(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-4)
lstm_model, history = train(lstm_model, train_dataloader, valid_dataloader, criterion, optimizer, device, 100, 'lstm_model')

class EmotionDatasetForMean(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        x = x.mean(axis=0)
        x = x.unsqueeze(0)
        y = self.label[index]
        return x, y

train_mean_dataset = EmotionDatasetForMean(train_audio, train_label)
train_mean_dataloader = DataLoader(train_mean_dataset, batch_size=batch_size, shuffle=True)
valid_mean_dataset = EmotionDatasetForMean(valid_audio, valid_label)
valid_mean_dataloader = DataLoader(valid_mean_dataset, batch_size=batch_size, shuffle=True)

input_size = train_mean_dataset[0][0].shape[-1]
hidden_size = 256
num_classes = 6
lstm_model = EmotionClassifierWithLSTM(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-4)
lstm_model, history = train(lstm_model, train_mean_dataloader, valid_mean_dataloader, criterion, optimizer, device, 100, 'mean_lstm_model')

class EmotionDatasetForDNN(Dataset):
    def __init__(self, data, label, axis=1):
        self.data = data
        self.label = label
        self.axis = axis

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        mean_x = x.mean(axis=self.axis)
        max_x = x.max(axis=self.axis)[0]
        min_x = x.min(axis=self.axis)[0]
        x = torch.cat([mean_x, max_x, min_x])
        y = self.label[index]
        return x, y

class EmotionClassifierWithDNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EmotionClassifierWithDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.fc2(out)  # 마지막 시점의 출력만 사용
        return out

train_dnn_dataset = EmotionDatasetForDNN(train_audio, train_label)
train_dnn_dataloader = DataLoader(train_dnn_dataset, batch_size=batch_size, shuffle=True)
valid_dnn_dataset = EmotionDatasetForDNN(valid_audio, valid_label)
valid_dnn_dataloader = DataLoader(valid_dnn_dataset, batch_size=batch_size, shuffle=True)

input_size = train_dnn_dataset[0][0].shape[-1]
hidden_size = 256
num_classes = 6
dnn_model = EmotionClassifierWithDNN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(dnn_model.parameters(), lr=1e-4)
dnn_model, history = train(dnn_model, train_dnn_dataloader, valid_dnn_dataloader, criterion, optimizer, device, 100, 'dnn_model')

# 2
train_dnn_dataset = EmotionDatasetForDNN(train_audio, train_label, axis=0)
train_dnn_dataloader = DataLoader(train_dnn_dataset, batch_size=batch_size, shuffle=True)
valid_dnn_dataset = EmotionDatasetForDNN(valid_audio, valid_label, axis=0)
valid_dnn_dataloader = DataLoader(valid_dnn_dataset, batch_size=batch_size, shuffle=True)

input_size = train_dnn_dataset[0][0].shape[-1]
hidden_size = 256
num_classes = 6
dnn_model_axis0 = EmotionClassifierWithDNN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(dnn_model_axis0.parameters(), lr=1e-4)
dnn_model_axis0, history = train(dnn_model_axis0, train_dnn_dataloader, valid_dnn_dataloader, criterion, optimizer, device, 100, 'dnn_model')
