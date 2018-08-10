import os
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import librosa.display as disp
import matplotlib.pyplot as plt
from hmmlearn import hmm


def load_data(path, sr=16000):
    base_path = "/home/hidehisa/hobby/pattern_recognition/sound_recognition" + \
                "/pyscripts"
    if os.getcwd() != base_path:
        os.chdir(base_path)

    x, fs = librosa.load(path, sr=44100)
    mfccs = librosa.feature.mfcc(x, sr=sr)
    return mfccs, sr


def hmm_fit(mfccs, train_length):
    model = hmm.GaussianHMM(n_components=3, n_iter=100)
    model.fit(mfccs[:train_length])
    return model


def main(path, sr, train_length):
    mfccs, fs = load_data(path, sr)
    print(f"MFCC shape: {mfccs.shape}")
    model = hmm_fit(mfccs.T, train_length)
    print("Fitting finished")
    preds = model.predict(mfccs.T[train_length:])
    print("Prediction finished")
    prediction = pd.DataFrame({
        'prediction': preds
    })
    prediction.to_csv('../raw_data/prediction.csv', index=False)


if __name__ == '__main__':
    path = '../raw_data/lab_meeting.wav'
    sr = 44100
    train_length = 500000
    main(path, sr, train_length)
