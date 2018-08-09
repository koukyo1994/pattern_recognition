import bnpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    hmodel, Rinfo = bnpy.run('../raw_data/artificial.csv',
                             'FiniteMixtureModel', 'Gauss', 'EM',
                             K=2, output_path='out/', nLap=500,
                             minLaps=50)

    min_all = hmodel.obsModel.get_mean_for_comp()[0].min()
    mu_all = hmodel.obsModel.get_mean_for_comp()[0] - min_all
    index_sort = np.sum(np.absolute(mu_all), axis=1).argsort()
    Data = pd.read_csv('../raw_data/artificial.csv')
    LP = hmodel.calc_local_params(Data)
    Z = LP['resp'].argmax(axis=1)

    Z_sort = np.zeros((50000,))
    Z_sort[Z == index_sort[0]] = 0
    Z_sort[Z == index_sort[1]] = 1
    plt.figure()
    plt.plot(Data['t'], Z_sort)
    plt.xlabel('time')
    plt.ylabel('x')
    plt.show()
