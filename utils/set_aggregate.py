import os
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt

rows = ['S1_r1', 'S1_r5', 'S1_r10', 'S1_r25', 'S1_r50', 'S1_p5', 'S1_p75', 'S1_p9', 'S2_means', 'S2_nearest', 'S2_random']
cols = ['r1', 'r5', 'r10', 'r25', 'r50', 'p5', 'p75', 'p9']

dir = '/data/nlp/vae/model/gmc/sets'
filenames = os.listdir(dir)
n = len(filenames)
forward_entailment_vals = defaultdict(list)
for idx, filename in enumerate(filenames):
    if idx % 100 == 0:
        print('{0} of {1}'.format(idx, n), end='\r')
    filepath = os.path.join(dir, filename)
    with open(filepath, 'r') as f:
        i = 0
        table = {}
        for row in rows:
            table[row] = defaultdict(int)
        for line in f:
            if i > len(rows):
                break
            if i > 0:
                vals = line.rstrip().split('\t')[1:]
                for j in range(len(cols)):
                    val = vals[j]
                    if ',' in val:
                        val = val.split(',')[0].replace('(','')
                    val = float(val)
                    table[rows[i-1]][cols[j]] = val
            i += 1
        for (row, col) in [(rows[i], cols[i]) for i in range(1,5)]:
            forward_entailment_vals[col].append(table[row][col])
        for row in rows[-3:]:
            for col in cols:
                forward_entailment_vals[row + '_' + col].append(table[row][col])

thresholds = np.arange(0., 1., .01)
plots = defaultdict(list)
for key in forward_entailment_vals.keys():
    for threshold in thresholds:
        sum = 0
        for val in forward_entailment_vals[key]:
            if val > threshold:
                sum += 1
        plots[key].append((threshold, sum / float(n)))

for key in plots.keys():
    x = [v[0] for v in plots[key]]
    y = [v[1] for v in plots[key]]
    plt.plot(x, y)
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.ylim([0, 1])
    with open(os.path.join('model/gmc/figs', key), 'wb+') as f:
        plt.savefig(f)
    plt.clf()
