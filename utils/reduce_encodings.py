import h5py
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input encodings path', type=str, required=True)
parser.add_argument('-o', '--output', help='output filename', type=str, required=True)
parser.add_argument('-n', '--n_components', help='number of dimensions to reduce to', type=int, default=25)
args = parser.parse_args()

with h5py.File(args.input) as f:
    with h5py.File(args.output, 'w') as o:
        encodings = f['encodings']
        filenames = f['filenames']
        pca = PCA(n_components=args.n_components)
        o.create_dataset('filenames', data=filenames)
        o.create_dataset('encodings', data=pca.fit_transform(encodings))
