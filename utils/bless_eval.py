from metrics import Metrics
import sys
sys.path.append('/data/nlp/bless')
from bless import Bless
import pickle
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='inut encodings path', type=str, default='model/gmc/encodings.hdf5')
parser.add_argument('-o', '--output', help='write detailed output to given file', type=str, default='model/gmc/tmp')
parser.add_argument('--dispersion', action='store_true')
parser.add_argument('--centroid', action='store_true')
parser.add_argument('--entropy', action='store_true')
parser.add_argument('--gaussian_rand', action='store_true')
parser.add_argument('--gaussian_dir', action='store_true')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

bless = Bless()
metrics = Metrics(args.input)
words = list(metrics.encoding_word_indices.keys())

if args.dispersion:
    correct = 0
    sum = 0
    for i in tqdm(range(len(bless.pairs))):
        w1, w2 = bless.pairs[i]
        if w1 not in words or w2 not in words:
            continue
        if metrics.dispersion(w2) > metrics.dispersion(w1):
            correct += 1
        sum += 1
    print('Dispersion: {0}'.format(correct / float(sum)))

if args.centroid:
    correct = 0
    sum = 0
    for i in tqdm(range(len(bless.pairs))):
        w1, w2 = bless.pairs[i]
        if w1 not in words or w2 not in words:
            continue
        if metrics.centroid(w2) > metrics.centroid(w1):
            correct += 1
        sum += 1
    print('Centroid: {0}'.format(correct / float(sum)))

if args.entropy:
    correct = 0
    sum = 0
    for i in tqdm(range(len(bless.pairs))):
        w1, w2 = bless.pairs[i]
        if w1 not in words or w2 not in words:
            continue
        if metrics.entropy(w2) > metrics.entropy(w1):
            correct += 1
        sum += 1
    print('Entropy: {0}'.format(correct / float(sum)))

if args.gaussian_rand:
    correct = 0
    sum = 0
    for i in tqdm(range(len(bless.pairs))):
        w1, w2 = bless.pairs[i]
        if w1 not in words or w2 not in words:
            continue
        gauss_w1 = metrics.gaussian_random(w1)
        gauss_w2 = metrics.gaussian_random(w2)
        if gauss_w1 == -1 or gauss_w2 == -1:
            continue
        if gauss_w2 > gauss_w1:
            correct += 1
        sum += 1
    print('Gaussian_random: {0}'.format(correct / float(sum)))

if args.gaussian_dir:
    correct = 0
    sum = 0
    with open(args.output, 'w+') as f:
        for i in tqdm(range(len(bless.pairs))):
            w1, w2 = bless.pairs[i]
            if w1 not in words or w2 not in words:
                continue
            w1_w2_p = metrics.gaussian_dir(w1, w2)
            w2_w1_p = metrics.gaussian_dir(w2, w1)
            if w1_w2_p == -1 or w2_w1_p == -1:
                continue
            f.write('{0}\n'.format('\t'.join(str(e) for e in [w1, w2, w1_w2_p, w2_w1_p, w2_w1_p - w1_w2_p, int(w2_w1_p > w1_w2_p)])))
            correct += w2_w1_p > w1_w2_p
            sum += 1
    print('Gaussian_dir: {0}'.format(correct / float(sum)))

if args.test:
    correct = 0
    sum = 0
    for i in tqdm(range(len(bless.pairs))):
        w1, w2 = bless.pairs[i]
        if w1 not in words or w2 not in words:
            continue
        correct += 0
        sum += 1
    print('test: {0}'.format(correct / float(sum)))
