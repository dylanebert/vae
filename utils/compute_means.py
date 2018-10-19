import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', help='path to input vectors', type=str, required=True)
parser.add_argument('--output_path', help='path to output means', type=str, required=True)
args = parser.parse_args()

with open(args.input_path, 'rb') as f:
    encodings = pickle.load(f)
print('Computing means')
means = {}
for label, entry in encodings.items():
    encodings = entry['encodings']
    means[label] = np.mean(encodings, axis=0).tolist()
with open(args.output_path, 'wb+') as f:
    pickle.dump(means, f)
