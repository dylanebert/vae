import pickle
import os
import numpy as np
import json
from scipy import spatial
from sklearn import preprocessing

def predict(exemplar_path, save_path):
    with open('model/gmc/encodings.p', 'rb') as f:
        encodings = pickle.load(f)

    with open(exemplar_path, 'rb') as f:
        exemplars = pickle.load(f)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    k = 0
    n = len(encodings)
    labels = np.array(list(exemplars.keys()))
    exemplar_vals = np.array(list(exemplars.values()))
    exemplar_vals_norm = preprocessing.normalize(exemplar_vals, norm='l2')
    tree_cos = spatial.KDTree(exemplar_vals_norm)
    tree_euc = spatial.KDTree(exemplar_vals)
    for label, entry in encodings.items():
        encodings = entry['encodings']
        filenames = entry['filenames']
        path = os.path.join(save_path, label + '.json')
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
                line = json.dumps({'label': label, 'filename': filenames[i], 'cos': spatial.distance.cosine(enc, exemplars[label]), 'euc': spatial.distance.euclidean(enc, exemplars[label]), 'predictions_cos': nearest_cos, 'predictions_euc': nearest_euc})
                with open(path, 'a+') as f:
                    f.write('{0}\n'.format(line))
            except:
                print('Failed on {0}'.format(label))


if __name__ == '__main__':
    #exemplar_path = 'model/gmc/means.p'
    exemplar_path = 'model/gmc/exemplars_random.p'

    #save_path = 'model/gmc/predictions/means'
    save_path = 'model/gmc/predictions/exemplars_random'

    predict(exemplar_path, save_path)
