import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance

import torch
import torchaudio

from textblob import TextBlob
from transformers import AutoProcessor, Wav2Vec2ForCTC

def apply_wav2vec(df, default_rate = 16000):
    df = df.copy()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.to(device)
    model.eval()

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        waveform, sample_rate = torchaudio.load(row['path'])
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=default_rate)

        inputs = processor(waveform[0], sampling_rate=default_rate, return_tensors="pt")
        array = inputs['input_values'].cpu().numpy().tolist()[0]
        df.at[idx,'array']=array

        with torch.no_grad():
            logits = model(**inputs.to(device)).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0].lower()
        transcription = ''.join(TextBlob(transcription).correct())
        df.loc[idx,'transcription']=transcription
    return df