import bnpy
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ndim = 20
    train_path = "../raw_data/train.wav"
    train_wav, _ = librosa.load(train_path, sr=44100)
    train = librosa.feature.mfcc(train_wav,
                                 sr=44100, n_mfcc=ndim)
    colnames = ['mfcc'+str(i) for i in range(train.shape[0])]
    df = pd.DataFrame(data=train.T, columns=colnames)
    df.to_csv('../raw_data/train_mfcc.csv', index=False)

    hmodel, Rinfo = bnpy.run('../raw_data/train_mfcc.csv',
                             'FiniteMixtureModel', 'Gauss', 'EM',
                             K=3, output_path='out/2/', nLap=300,
                             minLaps=10)
    plt.plot(np.arange(0,ndim),
             hmodel.obsModel.get_mean_for_comp(0),
             label="Cluster 0")
    plt.plot(np.arange(0,ndim),
             hmodel.obsModel.get_mean_for_comp(1),
             label="Cluster 1")
    plt.plot(np.arange(0,ndim),
             hmodel.obsModel.get_mean_for_comp(2),
             label="Cluster 2")
    plt.xlabel('Feat.')
    plt.ylabel('Norm. Pow')
    plt.ylim(-100, 180)
    plt.savefig('figure/normpVSfeat.png')
