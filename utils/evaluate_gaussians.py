import h5py
import numpy as np
from scipy.stats import multivariate_normal
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

w1 = 'dog'
w2 = 'animal'

with h5py.File('model/gmc/gaussians_2d.hdf5') as f:
    stacked_w1 = f[w1]
    stacked_w2 = f[w2]
    w1_mean = stacked_w1[0]
    w2_mean = stacked_w2[0]
    w1_cov = stacked_w1[1:]
    w2_cov = stacked_w2[1:]

    w1_normal = multivariate_normal(w1_mean, w1_cov)
    w2_normal = multivariate_normal(w2_mean, w2_cov)

    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))

    plt.figure()
    plt.subplot(121)
    plt.contourf(x, y, w1_normal.pdf(pos))
    plt.title(w1)
    plt.subplot(122)
    plt.contourf(x, y, w2_normal.pdf(pos))
    plt.title(w2)
    plt.show()
