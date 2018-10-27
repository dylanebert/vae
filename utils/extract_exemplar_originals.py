import pickle
from shutil import copy
import os

with open('/data/nlp/vae/model/gmc/encodings/all.p', 'rb') as f:
    encodings = pickle.load(f)

def extract(exemplars, method):
    for label, exemplar in exemplars.items():
        encs = encodings[label]['encodings']
        filenames = encodings[label]['filenames']
        idx = -1
        for i in range(len(encs)):
            if str(encs[i]) == str(exemplar):
                idx = i
        if idx == -1:
            print('Couldn\'t find {0}'.format())
        filename = filenames[idx]
        original_path = os.path.join('/data/nlp/gmc/train', filename)
        target_path = os.path.join('/data/nlp/vae/model/gmc/images/original', method, label + '.jpg')
        copy(original_path, target_path)

if __name__ == '__main__':
    with open('/data/nlp/vae/model/gmc/encodings/exemplars_random.p', 'rb') as f: #rnearest/random
        exemplars = pickle.load(f)
    extract(exemplars, 'random') #nearest/random
