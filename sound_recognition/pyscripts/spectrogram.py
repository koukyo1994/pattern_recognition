import os
import librosa
import librosa.display
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(path, duration=1000):
    base_path = os.environ['HOME'] + '/hobby/pattern_recognition/sound_recognition/scripts'
    if os.getcwd() != base_path:
        os.chdir(base_path)

    data, sr = librosa.load(path, sr=44100, duration=duration)
    return data, sr


if __name__ == '__main__':
    path = '../raw_data/lab_meeting.wav'
    data, sr = load_data(path, duration=50)
    plt.figure()
    librosa.display.waveplot(data, sr=sr)
    plt.show()
