from metrics import Metrics
import sys
sys.path.append('/data/nlp/bless')
from bless import Bibless
import pickle
from tqdm import tqdm
import argparse
from scipy.spatial.distance import cosine

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='inut encodings path', type=str, default='model/gmc/encodings.hdf5')
parser.add_argument('-o', '--output', help='write detailed output to given file', type=str, default='model/gmc/tmp')
parser.add_argument('--dispersion', action='store_true')
parser.add_argument('--centroid', action='store_true')
parser.add_argument('--entropy', action='store_true')
parser.add_argument('--gaussian_rand', action='store_true')
parser.add_argument('--gaussian_dir', action='store_true')
parser.add_argument('--beta', help='override beta argument for gaussian_dir', type=float, default=1e-7)
parser.add_argument('--gamma', help='override gamma argument for gaussian_rand', type=float, default=1e-13)
args = parser.parse_args()

bibless = Bibless()
metrics = Metrics(args.input)
words = list(metrics.encoding_word_indices.keys())

alpha = .02
theta = .25

if args.dispersion:
    correct = 0
    sum = 0
    for i in tqdm(range(len(bibless.pairs))):
        w1, w2 = bibless.pairs[i]
        true_label = bibless.truth_vals[i]
        if w1 not in words or w2 not in words:
            continue
        entails = 0
        cossim = 1 - cosine(metrics.mean(w1), metrics.mean(w2))
        if cossim > theta:
            s = 1 - ((metrics.dispersion(w1) + alpha) / metrics.dispersion(w2))
            if s > 0:
                entails = 1
            else:
                entails = -1
        correct += int(entails == true_label)
        sum += 1
    print('Dispersion: {0}'.format(correct / float(sum)))

if args.centroid:
    correct = 0
    sum = 0
    for i in tqdm(range(len(bibless.pairs))):
        w1, w2 = bibless.pairs[i]
        true_label = bibless.truth_vals[i]
        if w1 not in words or w2 not in words:
            continue
        entails = 0
        cossim = 1 - cosine(metrics.mean(w1), metrics.mean(w2))
        if cossim > theta:
            s = 1 - ((metrics.centroid(w1) + alpha) / metrics.centroid(w2))
            if s > 0:
                entails = 1
            else:
                entails = -1
        correct += int(entails == true_label)
        sum += 1
    print('Centroid: {0}'.format(correct / float(sum)))

if args.entropy:
    correct = 0
    sum = 0
    for i in tqdm(range(len(bibless.pairs))):
        w1, w2 = bibless.pairs[i]
        true_label = bibless.truth_vals[i]
        if w1 not in words or w2 not in words:
            continue
        entails = 0
        cossim = 1 - cosine(metrics.mean(w1), metrics.mean(w2))
        if cossim > theta:
            s = 1 - ((metrics.entropy(w1) + alpha) / metrics.entropy(w2))
            if s > 0:
                entails = 1
            else:
                entails = -1
        correct += int(entails == true_label)
        sum += 1
    print('Entropy: {0}'.format(correct / float(sum)))

if args.gaussian_rand:
    correct = 0
    sum = 0
    for i in tqdm(range(len(bibless.pairs))):
        w1, w2 = bibless.pairs[i]
        true_label = bibless.truth_vals[i]
        if w1 not in words or w2 not in words:
            continue
        entails = 0
        cossim = 1 - cosine(metrics.mean(w1), metrics.mean(w2))
        if cossim > theta:
            s = metrics.gaussian_random_pair(w1, w2)
            if s < args.gamma:
                entails = 1
            else:
                entails = -1
        correct += int(entails == true_label)
        sum += 1
    print('Gaussian_rand: {0}'.format(correct / float(sum)))

if args.gaussian_dir:
    correct = 0
    sum = 0
    with open(args.output, 'w+') as f:
        for i in tqdm(range(len(bibless.pairs))):
            w1, w2 = bibless.pairs[i]
            true_label = bibless.truth_vals[i]
            if w1 not in words or w2 not in words:
                continue
            w1_w2_p = metrics.gaussian_dir(w1, w2)
            w2_w1_p = metrics.gaussian_dir(w2, w1)
            if w1_w2_p == -1 or w2_w1_p == -1:
                continue
            if w1_w2_p > args.beta and w2_w1_p > args.beta:
                if w1_w2_p > w2_w1_p:
                    entails = 1
                else:
                    entails = -1
            elif w1_w2_p > args.beta:
                entails = 1
            elif w2_w1_p > args.beta:
                entails = -1
            else:
                entails = 0
            #print(w1, w2, w1_w2_p, true_label, entails)
            f.write('{0}\t{1}\n'.format(str(entails), str(true_label)))
            correct += int(entails == true_label)
            sum += 1
    print('Gaussian_dir: {0}'.format(correct / float(sum)))
