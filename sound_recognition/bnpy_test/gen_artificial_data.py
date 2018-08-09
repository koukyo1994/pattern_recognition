import numpy as np
import pandas as pd
import numpy.random as random
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    t = np.linspace(0.0, 500.01, 50000)
    x = np.sin(0.1 * t + 0.01) + 3.0 + random.normal(0.0, 1.0, (50000, ))
    y = 0.1
    x[30000:] += y

    plt.plot(t, x)
    plt.xlabel('time [t]')
    plt.ylabel('x value')
    plt.savefig('figure/artificial.png')

    df = pd.DataFrame({
        'x': x,
        't': t
    })
    df.to_csv('../raw_data/artificial.csv', index=False)
