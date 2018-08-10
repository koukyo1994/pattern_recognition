import librosa


if __name__ == '__main__':
    data, sr = librosa.load('../raw_data/lab_meeting.wav', sr=44100)
    train = data[:26460000]
    test = data[26460000:52920000]
    librosa.
