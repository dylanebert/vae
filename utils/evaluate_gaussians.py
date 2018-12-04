import h5py
import numpy as np
from scipy.stats import multivariate_normal
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

w1 = 'animal'
w2 = 'dog'

with open('model/gmc/encoding_word_indices.p', 'rb') as f:
    encoding_word_indices = pickle.load(f)

with h5py.File('model/gmc/encodings_reduced.hdf5') as f:
    i, n = encoding_word_indices[w1]
    w1_encodings = f['encodings'][i:i+n]

    i, n = encoding_word_indices[w2]
    w2_encodings = f['encodings'][i:i+n]

    w1_mean = np.mean(w1_encodings, axis=0)
    w1_cov = np.cov(w1_encodings.T)
    w1_normal = multivariate_normal(w1_mean, w1_cov)

    w2_mean = np.mean(w2_encodings, axis=0)
    w2_cov = np.cov(w2_encodings.T)
    w2_normal = multivariate_normal(w2_mean, w2_cov)

    w1_predictions = np.zeros((len(w1_encodings), 2))
    for i, val in enumerate(w1_encodings):
        w1_predictions[i][0] = w1_normal.pdf(val)
        w1_predictions[i][1] = w2_normal.pdf(val)

    print(w1_normal.pdf(w1_mean))
    for item in np.mean(w1_predictions, axis=0):
        print(item)
