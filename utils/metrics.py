import os
import h5py
import pickle
from itertools import combinations
from scipy.spatial.distance import cosine
from scipy.stats import multivariate_normal
import numpy as np

class Metrics():
    def __init__(self, encodings_path):
        self.encodings_path = encodings_path
        with open('model/gmc/encoding_word_indices.p', 'rb') as f:
            self.encoding_word_indices = pickle.load(f)

    def mean(self, word):
        i, n = self.encoding_word_indices[word]
        with h5py.File(self.encodings_path) as f:
            encodings = f['encodings'][i:i+n]
            return np.mean(encodings, axis=0)

    def dispersion(self, word):
        i, n = self.encoding_word_indices[word]
        sum_cos = 0
        with h5py.File(self.encodings_path) as f:
            encodings = f['encodings'][i:i+n]
            for x, y in combinations(encodings, 2):
                sum_cos += cosine(x, y)
        dispersion = (2 / (n * (n - 1))) * sum_cos
        return dispersion

    def centroid(self, word):
        i, n = self.encoding_word_indices[word]
        sum_dist = 0
        with h5py.File(self.encodings_path) as f:
            encodings = f['encodings'][i:i+n]
            mean = np.mean(encodings, axis=0)
            for encoding in encodings:
                sum_dist += cosine(encoding, mean)
        centroid = sum_dist / float(n)
        return centroid

    def entropy(self, word):
        i, n = self.encoding_word_indices[word]
        sum = 0
        mean = self.mean(word)
        for j in range(len(mean)):
            p = np.abs(mean[j] / float(n))
            sum += p * np.log2(p)
        return -sum

    def gaussian_random(self, word):
        i, n = self.encoding_word_indices[word]
        p_sum = 0
        with h5py.File(self.encodings_path) as f:
            encodings = f['encodings'][i:i+n]
            l = encodings.shape[1]
            if n < l:
                return -1
            mean = np.mean(encodings, axis=0)
            cov = np.cov(encodings.T)
            normal = multivariate_normal(mean=mean, cov=cov)
            for v in np.random.normal(size=(1000, l)):
                p = normal.pdf(v)
                p_sum += p
            return p_sum

    def gaussian_dir(self, w1, w2): #returns probability that centroid of w2 is in distribution of w1
        w1_i, w1_n = self.encoding_word_indices[w1]
        w2_i, w2_n = self.encoding_word_indices[w2]
        with h5py.File(self.encodings_path) as f:
            w1_encodings = f['encodings'][w1_i:w1_i+w1_n]
            w2_encodings = f['encodings'][w2_i:w2_i+w2_n]
            l = w1_encodings.shape[1]
            if w1_n < l or w2_n < l:
                return -1
            w1_mean = np.mean(w1_encodings, axis=0)
            w2_mean = np.mean(w2_encodings, axis=0)
            w1_cov = np.cov(w1_encodings.T)
            w1_normal = multivariate_normal(mean=w1_mean, cov=w1_cov)
            return w1_normal.pdf(w2_mean)

if __name__ == '__main__':
    metrics = Metrics('model/gmc/encodings.hdf5')
    print(metrics.dispersion('dog'))
    print(metrics.dispersion('animal'))
    print(metrics.dispersion('person'))
    print(metrics.dispersion('entity'))
    print(metrics.centroid('dog'))
    print(metrics.centroid('animal'))
    print(metrics.centroid('person'))
    print(metrics.centroid('entity'))
    print(metrics.entropy('dog'))
    print(metrics.entropy('animal'))
    print(metrics.entropy('person'))
    print(metrics.entropy('entity'))
