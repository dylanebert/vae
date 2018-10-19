import random
import pickle
import numpy as np
from scipy.spatial.distance import euclidean
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', help='path to model (must contain encodings/all.p and encodings/means.p)', type=str, required=True)
args = parser.parse_args()

encodings_path = os.path.join(args.model_path, 'encodings', 'all.p')
means_path = os.path.join(args.model_path, 'encodings', 'means.p')
exemplars_nearest_path = os.path.join(args.model_path, 'encodings', 'exemplars_nearest.p')
exemplars_random_path = os.path.join(args.model_path, 'encodings', 'exemplars_random.p')

with open(encodings_path, 'rb') as f:
    encodings = pickle.load(f)
with open(means_path, 'rb') as f:
    means = pickle.load(f)

exemplars_nearest = {}
exemplars_random = {}

for label, entry in encodings.items():
    mean = means[label]
    encodings = entry['encodings']
    dists = [euclidean(v, mean) for v in encodings]
    idx_sort = np.argsort(dists)
    idx_nearest = idx_sort[0]
    idx_random = random.choice(idx_sort)
    exemplars_nearest[label] = encodings[idx_nearest]
    exemplars_random[label] = encodings[idx_random]

with open(exemplars_nearest_path, 'wb+') as f:
    pickle.dump(exemplars_nearest, f)
with open(exemplars_random_path, 'wb+') as f:
    pickle.dump(exemplars_random, f)
