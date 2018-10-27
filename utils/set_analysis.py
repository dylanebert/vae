import os
import csv
from collections import defaultdict
import json
import pickle

with open('data/gmc_words', 'r') as f:
    gmc_words = f.read().splitlines()
with open('data/wordnet_gmc_filtered_pairs', 'r') as f:
    wordnet_gmc_pairs = [tuple(line.rstrip().split(' ')) for line in f.read().splitlines()]

classifier_membership_dir = '/home/dylan/Documents/inception/model/gmc/membership'
classifier_methods = ['r1', 'r5', 'r10', 'r25', 'r50', 'p5', 'p75', 'p9']

vae_set_membership_dir = '/data/nlp/vae/model/gmc/membership/knearest'
vae_set_methods = ['means', 'exemplars_nearest', 'exemplars_random']

exemplar_dir = '/home/dylan/Documents/inception/model/gmc/exemplars'
exemplar_subdirs = ['original/nearest', 'original/random', 'reconstruction/nearest', 'reconstruction/means', 'reconstruction/random']

cols = classifier_methods
rows = ['S1_r1', 'S1_r5', 'S1_r10', 'S1_r25', 'S1_r50', 'S1_p5', 'S1_p75', 'S1_p9', 'S2_means', 'S2_nearest', 'S2_random'] + exemplar_subdirs

n = len(wordnet_gmc_pairs)
for i, (w1, w2) in enumerate(wordnet_gmc_pairs):
    print('{0} of {1}: {2}, {3}'.format(i+1, n, w1, w2))
    table = {}
    for row in rows:
        table[row] = defaultdict(int)
    for set, dir, methods in [('S1', classifier_membership_dir, classifier_methods), ('S2', vae_set_membership_dir, vae_set_methods)]:
        for row_method in methods:
            w1_filepath = os.path.join(dir, row_method, w1)
            w1_members = []
            row_header = '{0}_{1}'.format(set, row_method.replace('exemplars_',''))
            if os.path.exists(w1_filepath):
                with open(w1_filepath, 'r') as f:
                    w1_members = f.read().splitlines()

                for col_method in classifier_methods:
                    w2_filepath = os.path.join(classifier_membership_dir, col_method, w2)
                    w2_members = []
                    precision = 0
                    recall = 0
                    col_header = col_method
                    if os.path.exists(w2_filepath):
                        with open(w2_filepath, 'r') as f:
                            w2_members = f.read().splitlines()

                        true_positives = [member for member in w1_members if member in w2_members]
                        precision = len(true_positives) / float(len(w1_members))
                        recall = len(true_positives) / float(len(w2_members))
                        true_subset = True
                        for member in w1_members:
                            if member not in w2_members:
                                true_subset = False

                    table[row_header][col_header] = (precision, recall, int(true_subset))
    for dir in exemplar_subdirs:
        row_header = dir
        with open(os.path.join(exemplar_dir, dir, w1), 'r') as f:
            labels = []
            p_vals = []
            for line in f:
                label, p_val = line.rstrip().split('\t')
                labels.append(label)
                p_vals.append(float(p_val))
            for r in [1, 5, 10, 25, 50]:
                col_header = 'r' + str(r)
                table[row_header][col_header] = int(w2 in labels[:r])
            for p in [.5, .75, .9]:
                col_header = 'p' + str(p * 100).replace('.','').replace('0','')
                table[row_header][col_header] = int(p_vals[labels.index(w2)] >= p)
    with open(os.path.join('/data/nlp/vae/model/gmc/sets', w1 + '_' + w2), 'w+') as f:
        f.write('\t'.join(['wordnet'] + cols) + '\n')
        for row in rows:
            f.write('\t'.join([row] + [str(table[row][col]).replace(' ','') for col in cols]) + '\n')
