import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    base_path = "/home/hidehisa/hobby/pattern_recognition/sound_recognition" + \
                "/scripts"
    if os.getcwd() != base_path:
        os.chdir(base_path)

    x, fs = librosa.load(path, sr=44100)
    mfccs = librosa.feature.mfcc(x, sr=fs)
    return mfccs


if __name__ == '__main__':
    path = "../raw_data/lab_meeting.wav"
    mfccs = load_data(path)
    
