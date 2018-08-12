import numpy as np
import pandas as pd
import librosa
import librosa.display
from hmmlearn import hmm
import matplotlib.pyplot as plt


def train_hmm(train):
    model = hmm.GaussianHMM(n_components=3, n_iter=100)
    model.fit(train)
    return model


def test_hmm(model, test):
    preds = model.predict(test)
    df = pd.DataFrame({'prediction': preds})
    df.to_csv('../raw_data/cutout.csv', index=False)


def main():
    print("Data Load")
    train_path = "../raw_data/train.wav"
    test_path = "../raw_data/test.wav"

    train_wav, _ = librosa.load(train_path, sr=44100)
    test_wav, _ = librosa.load(test_path, sr=44100)

    print("Convert into MFCC")
    train = librosa.feature.mfcc(train_wav, sr=44100)
    test = librosa.feature.mfcc(test_wav, sr=44100)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(train, sr=44100, x_axis='time')
    plt.title('Train data MFCC')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    librosa.display.specshow(test, sr=44100, x_axis='time')
    plt.title('Test data MFCC')
    plt.colorbar()
    plt.savefig('figure/mfcc.png')

    print("Fitting")
    model = train_hmm(train.T)

    print("Prediction")
    test_hmm(model, test.T)


if __name__ == '__main__':
    main()
