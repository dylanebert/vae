import pickle
import os
from scipy.spatial.distance import cosine, euclidean

with open('model/gmc/encodings.p', 'rb') as f:
    encodings = pickle.load(f)
with open('model/gmc/means.p', 'rb') as f:
    means = pickle.load(f)

k = 0
n = len(encodings)
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
            nearest_cos = list(dict(sorted(means.items(), key=lambda x: cosine(enc, x[1]))[:100]).keys())
            nearest_euc = list(dict(sorted(means.items(), key=lambda x: euclidean(enc, x[1]))[:100]).keys())
            line = json.dumps({'label': label, 'filename': filenames[i], 'cos': cosine(enc, means[label]), 'euc': euclidean(enc, means[label]), 'predictions_cos': nearest_cos, 'predictions_euc': nearest_euc})
            with open(path, 'a+') as f:
                f.write('{0}\n'.format(line))
        except:
            print('Failed on {0}'.format(label))
