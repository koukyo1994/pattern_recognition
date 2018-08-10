import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path, sr=44100):
    base_path = "/home/hidehisa/hobby/pattern_recognition/sound_recognition" + \
                "/pyscripts"
    if os.getcwd() != base_path:
        os.chdir(base_path)

    x, fs = librosa.load(path, sr=44100)
    mfccs = librosa.feature.mfcc(x, sr=sr)
    return mfccs, sr


if __name__ == '__main__':
    path = "../raw_data/lab_meeting.wav"
    mfccs, sr = load_data(path, 44100)
    print(f"MFCC shape: {mfccs.shape}")
    librosa.display.specshow(mfccs[:, :10], sr=sr, x_axis='time')
    plt.show()
