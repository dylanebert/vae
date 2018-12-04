import h5py
import pickle
from tqdm import tqdm
import os

indices = {}
with h5py.File('model/gmc/encodings.hdf5') as f:
    filenames = f['filenames']
    current_word = None
    start_idx = 0
    n = 0
    for i in tqdm(range(len(filenames))):
        word = os.path.split(filenames[i])[0].decode('utf-8')
        if not word == current_word:
            if n > 0:
                indices[word] = (start_idx, n)
                n = 0
            start_idx = i
            current_word = word
        n += 1

with open('model/gmc/encoding_word_indices.p', 'wb+') as f:
    pickle.dump(indices, f)
