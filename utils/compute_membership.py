import os
import json
from collections import defaultdict

with open('data/gmc_words', 'r') as f:
    gmc_words = f.read().splitlines()

s2_membership = defaultdict(dict)
for method in ['means', 'exemplars_nearest', 'exemplars_random']:
    print('Collecting s2 set membership: {0}'.format(method))
    vae_predictions_dir = os.path.join('model/gmc/predictions', method)
    vae_prediction_files = os.listdir(vae_predictions_dir)
    for filename in vae_prediction_files:
        filepath = os.path.join(vae_predictions_dir, filename)
        with open(filepath, 'rb') as f:
            for line in f:
                line = json.loads(line)
                for word in line['predictions_euc'][:50]:
                    if word not in s2_membership[method]:
                        s2_membership[method][word] = []
                    s2_membership[method][word].append(line['filename'])

membership_dir = 'analysis/membership'
for method in s2_membership.keys():
    if not os.path.exists(os.path.join(membership_dir, method)):
        os.makedirs(os.path.join(membership_dir, method))
    for word in s2_membership[method].keys():
        with open(os.path.join(membership_dir, method, word), 'w+') as f:
            f.write('\n'.join(s2_membership[method][word]))
