import os
import librosa
import numpy as np
import pandas as pd
from hmmlearn import hmm


def hmm_fit(train_data):
    model = hmm.GaussianHMM(n_components=7, n_iter=500)
    model.fit(train_data)
    return model


def hmm_predict(model, test_data):
    preds = model.predict(test_data)
    df = pd.DataFrame({'prediction': preds})
    df.to_csv('../raw_data/pred_5-10.csv', index=False)


def main(path, train_length, test_length):
    print("Load data")
    train, sr = librosa.load(path,
                         sr=44100,
                         duration=train_length)
    test, sr = librosa.load(path,
                            sr=44100,
                            offset=train_length,
                            duration=test_length)
    print("Get MFCC")
    mfccs_train = librosa.feature.mfcc(train, sr=44100)
    mfccs_test = librosa.feature.mfcc(test, sr=44100)
    print("HMM train")
    model = hmm_fit(mfccs_train.T)
    print("HMM prediction")
    hmm_predict(model, mfccs_test.T)


if __name__ == '__main__':
    path = "../raw_data/lab_meeting.wav"
    main(path, 300, 300)
