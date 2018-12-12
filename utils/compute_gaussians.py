import h5py
import numpy as np
from scipy.stats import multivariate_normal
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input encodings', type=str, required=True)
parser.add_argument('-o', '--output', help='path to output gaussians', type=str, required=True)
parser.add_argument('-n', '--dimensions', help='specify number of dimensions', type=int, default=25)
args = parser.parse_args()

with open('model/gmc/encoding_word_indices.p', 'rb') as f:
    encoding_word_indices = pickle.load(f)

with h5py.File(args.input) as f:
    with h5py.File(args.output, 'w') as o:
        for word, (i, n) in encoding_word_indices.items():
            if n < args.dimensions: #not enough instances to compute covariance matrix
                continue
            encodings = f['encodings'][i:i+n]
            mean = np.mean(encodings, axis=0)
            cov = np.cov(encodings.T)
            stacked = np.concatenate(([mean], cov), axis=0)
            o.create_dataset(word, data=stacked)
            print('Saved {0}'.format(word))
