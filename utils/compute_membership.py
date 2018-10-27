import os
import json
from collections import defaultdict
import pickle
from scipy.spatial.distance import euclidean
import sys

def k_nearest(words):
    membership = defaultdict(dict)
    for method in ['means', 'exemplars_nearest', 'exemplars_random']:
        print('Collecting nearest K set membership: {0}'.format(method))
        vae_predictions_dir = os.path.join('/data/nlp/vae/model/gmc/predictions', method)
        vae_prediction_files = os.listdir(vae_predictions_dir)
        for filename in vae_prediction_files:
            filepath = os.path.join(vae_predictions_dir, filename)
            with open(filepath, 'rb') as f:
                for line in f:
                    line = json.loads(line)
                    for word in line['predictions_euc'][:50]:
                        if word not in membership[method]:
                            membership[method][word] = []
                        membership[method][word].append(line['filename'])

    membership_dir = '/data/nlp/vae/model/gmc/membership/knearest'
    for method in membership.keys():
        if not os.path.exists(os.path.join(membership_dir, method)):
            os.makedirs(os.path.join(membership_dir, method))
        for word in words:
            if word not in membership[method]:
                print('Couldn\'t find {0}'.format(word))
            with open(os.path.join(membership_dir, method, word), 'w+') as f:
                f.write('\n'.join(membership[method][word]))

def threshold(words):
    MAX_DIST = 2.8
    membership = defaultdict(dict)

    with open('/data/nlp/vae/model/gmc/encodings/all.p', 'rb') as f:
        encoding_dict = pickle.load(f)
        encoding_filename_dict = {}
        encodings = []
        for label, entry in encoding_dict.items():
            for encoding, filename in zip(entry['encodings'], entry['filenames']):
                encodings.append(encoding)
                encoding_filename_dict[tuple(encoding)] = filename

    for method in ['means', 'exemplars_nearest', 'exemplars_random']:
        print('Collecting threshold membership: {0}'.format(method))
        membership_dir = '/data/nlp/vae/model/gmc/membership/threshold'
        if not os.path.exists(os.path.join(membership_dir, method)):
            os.makedirs(os.path.join(membership_dir, method))
        with open(os.path.join('/data/nlp/vae/model/gmc/encodings', method + '.p'), 'rb') as f:
            exemplars = pickle.load(f)
            n = len(words)
            for i, word in enumerate(words):
                print('{0} of {1}'.format(i+1, n), end='\r')
                if not word in exemplars:
                    print('Couldn\'t find {0}'.format(word))
                    continue
                exemplar = exemplars[word]
                neighbors = []
                for encoding in encodings:
                    if euclidean(exemplar, encoding) < MAX_DIST:
                        neighbors.append(encoding_filename_dict[tuple(encoding)])
                with open(os.path.join(membership_dir, method, word), 'w+') as f:
                    f.write('\n'.join(neighbors))

if __name__ == '__main__':
    if not len(sys.argv) == 2:
        sys.exit('Invalid args. Use compute_membership.py [input_words_file]')
    input_filename = sys.argv[1]
    with open(input_filename, 'r') as f:
        words = f.read().splitlines()

    #k_nearest(words)
    #threshold(words)
    exemplars(words)
