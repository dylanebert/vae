import os
import csv
from collections import defaultdict
import json
import pickle

with open('data/gmc_words', 'r') as f:
    gmc_words = f.read().splitlines()
with open('data/wordnet_gmc_filtered_pairs', 'r') as f:
    wordnet_gmc_pairs = [tuple(line.rstrip().split(' ')) for line in f.read().splitlines()]

def get_s1_membership():
    s1_membership = defaultdict(dict)
    classifier_membership_dir = '/home/dylan/Documents/inception/model/gmc/membership'
    for method in ['r1', 'r5', 'r10', 'r25', 'r50', 'p5', 'p75', 'p9']:
        print('Collecting s1 set membership: {0}'.format(method))
        for word in gmc_words:
            filepath = os.path.join(classifier_membership_dir, method, word)
            if os.path.exists(filepath):
                with open(os.path.join(classifier_membership_dir, method, word), 'r') as f:
                    s1_membership[method][word] = f.read().splitlines()
    return s1_membership

def get_s2_membership():
    
    return s2_membership

if __name__ == '__main__':
    #s1_membership = get_s1_membership()
    #s2_membership = get_s2_membership()
