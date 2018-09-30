import numpy as np
import pickle
import os
import json
from config import Config
from vae import VAE
from data_generator import DataGenerator
from scipy.spatial.distance import cosine, euclidean

with open('model/gmc/means.p', 'rb') as f:
    means = pickle.load(f)

config = Config()
config.__dict__ = json.loads(open('model/gmc/config.json', 'r').read())

test_generator = DataGenerator(config.test_path, config.image_size, config.batch_size)

vae = VAE(config)
encodings = vae.encoder.predict_generator(test_generator, verbose=1)
print(encodings.shape)

for label, encs in encodings.items():
    break
    for i, enc in enumerate(encs):
        nearest = list(dict(sorted(means.items(), key=lambda x: cosine(enc, x[1]))[:100]).keys())
        line = json.dumps({'label': label, 'cos': cosine(enc, means[label]), 'euc': euclidean(enc, means[label]), 'predictions': nearest})
        with open(os.path.join('model/gmc/predictions', label + '.json'), 'a+') as f:
            f.write('{0}\n'.format(line))
