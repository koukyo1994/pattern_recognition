import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path, sr=16000):
    base_path = "/home/hidehisa/hobby/pattern_recognition/sound_recognition" + \
                "/scripts"
    if os.getcwd() != base_path:
        os.chdir(base_path)

    x, fs = librosa.load(path, sr=44100)
    mfccs, sr = librosa.feature.mfcc(x, sr=sr)
    return mfccs, sr


if __name__ == '__main__':
    path = "../raw_data/lab_meeting.wav"
    mfccs,sr = load_data(path)
    librosa.display.specshow(mfccs[:, :10], sr=sr, x_axis='time')
    plt.show()
