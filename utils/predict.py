import pickle
import os
import numpy as np
import json
from scipy import spatial
from sklearn import preprocessing

with open('model/gmc/encodings.p', 'rb') as f:
    encodings = pickle.load(f)
with open('model/gmc/means.p', 'rb') as f:
    means = pickle.load(f)

k = 0
n = len(encodings)
labels = np.array(list(means.keys()))
mean_vals = np.array(list(means.values()))
mean_vals_norm = preprocessing.normalize(mean_vals, norm='l2')
tree_cos = spatial.KDTree(mean_vals_norm)
tree_euc = spatial.KDTree(mean_vals)
for label, entry in encodings.items():
    encodings = entry['encodings']
    filenames = entry['filenames']
    path = os.path.join('model/gmc/predictions', label + '.json')
    if os.path.exists(path):
        os.remove(path)
    k += 1
    print('{0} of {1}'.format(k, n), end='\r')
    for i, enc in enumerate(encodings):
        try:
            cos_dists, cos_idx = tree_cos.query(enc, 50)
            nearest_cos = [labels[i] for i in cos_idx]
            euc_dists, euc_idx = tree_euc.query(enc, 50)
            nearest_euc = [labels[i] for i in euc_idx]
            line = json.dumps({'label': label, 'filename': filenames[i], 'cos': spatial.distance.cosine(enc, means[label]), 'euc': spatial.distance.euclidean(enc, means[label]), 'predictions_cos': nearest_cos, 'predictions_euc': nearest_euc})
            with open(path, 'a+') as f:
                f.write('{0}\n'.format(line))
        except:
            print('Failed on {0}'.format(label))
