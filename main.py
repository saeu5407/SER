import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, Wav2Vec2Processor
from datasets import load_dataset, load_metric
import torch
import torchaudio

import numpy as np

from tqdm.auto import tqdm
from scipy.spatial import distance
from textblob import TextBlob
from transformers import AutoProcessor, Wav2Vec2ForCTC

from src.preproc import apply_wav2vec


# model setting
model_name_or_path = "facebook/wav2vec2-base-960h" # jonatasgrosman/wav2vec2-large-xlsr-53-english 컴퓨터 성능좋으면 이거하자
pooling_mode = "mean"

# config
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=len(train.label.unique()),
    finetuning_task="wav2vec2_clf",
)
setattr(config, 'pooling_mode', pooling_mode) # 객체 내부의 속성을 변경해주는 파이썬 내장 함수로 config 파일에 pooling_mode를 추가한 것

processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
target_sampling_rate = processor.feature_extractor.sampling_rate

# Loading the created dataset using datasets

data_files = {
    "train": f"{dataset_path}/train_split.csv",
    "valid": f"{dataset_path}/valid_split.csv",
}
dataset = load_dataset("csv", data_files=data_files,)
train_dataset = dataset["train"]
eval_dataset = dataset["valid"]

print(train_dataset)
print(eval_dataset)

input_column = "path"
output_column = "label"

# 기존 오디오를 불러온 후, 모델의 sampling rate에 맞추어 리샘플링 하는 함수
def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

# Process 사이클을 돌리는 함수
def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    result = processor(speech_list, sampling_rate=target_sampling_rate)
    result['label'] = list(examples[output_column])

    return result

# 데이터셋에 대해 Process 사이클을 돌려서 전처리
train_dataset = train_dataset.map(
    preprocess_function,
    batch_size=100,
    batched=True,
    num_proc=4
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batch_size=100,
    batched=True,
    num_proc=4
)