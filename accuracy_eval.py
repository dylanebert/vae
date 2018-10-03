import numpy as np
import pickle
import os
import json
from scipy.spatial.distance import cosine, euclidean

def predict(method):
    with open('model/gmc/means.p', 'rb') as f:
        means = pickle.load(f)
    with open('model/gmc/test_encodings.p', 'rb') as f:
        encodings = pickle.load(f)

    k = 0
    n = len(encodings)
    for label, encs in encodings.items():
        k += 1
        print('{0} of {1}'.format(k, n), end='\r')
        for i, enc in enumerate(encs):
            if not os.path.exists(os.path.join('model/gmc/predictions', label + '.json')):
                try:
                    if method == 'cos':
                        nearest = list(dict(sorted(means.items(), key=lambda x: cosine(enc, x[1]))[:100]).keys())
                    else:
                        nearest = list(dict(sorted(means.items(), key=lambda x: euclidean(enc, x[1]))[:100]).keys())
                    line = json.dumps({'label': label, 'cos': cosine(enc, means[label]), 'euc': euclidean(enc, means[label]), 'predictions': nearest})
                    with open(os.path.join('model/gmc/predictions', label + '.json'), 'a+') as f:
                        f.write('{0}\n'.format(line))
                except:
                    print('Failed on {0}'.format(label))

if __name__ == '__main__':
    #predict('cos')
    predict('euc')
