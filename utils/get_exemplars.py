import json
import random
import pickle
import numpy as np
from scipy.spatial.distance import euclidean

with open('model/gmc/encodings.p', 'rb') as f:
    encodings = pickle.load(f)

with open('model/gmc/means.p', 'rb') as f:
    means = pickle.load(f)

with open('exemplars_nearest', 'w+') as f:
    with open('exemplars_random', 'w+') as g:
        for label, entry in encodings.items():
            mean = means[label]
            encodings = entry['encodings']
            filenames = entry['filenames']
            dists = [euclidean(v, mean) for v in encodings]
            idx_sort = np.argsort(dists)
            idx_nearest = idx_sort[0]
            idx_random = random.choice(idx_sort)
            exemplar_nearest = (filenames[idx_nearest], dists[idx_nearest])
            exemplar_random = (filenames[idx_random], dists[idx_random])
            f.write('{0}\t{1}\t{2}\n'.format(label, filenames[idx_nearest], dists[idx_nearest]))
            g.write('{0}\t{1}\t{2}\n'.format(label, filenames[idx_random], dists[idx_random]))
