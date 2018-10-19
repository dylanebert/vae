import json
import os
from collections import defaultdict
from itertools import product

with open('gmc_words', 'r') as f:
    gmc_words = f.read().splitlines()
with open('wordnet_gmc_filtered_pairs', 'r') as f:
    wordnet_gmc_pairs = [tuple(line.rstrip().split(' ')) for line in f]
wordnet_words = list(set([pair[0] for pair in wordnet_gmc_pairs] + [pair[1] for pair in wordnet_gmc_pairs]))
words_to_evaluate = [word for word in gmc_words if word in wordnet_words]

entailment_path = '/data/entailment'
vae_prediction_paths = {'means': 'model/gmc/predictions/means', 'exemplars_nearest': 'model/gmc/predictions/exemplars_nearest', 'exemplars_random': 'model/gmc/predictions/exemplars_random'}
classifier_prediction_path = '/home/dylan/Documents/inception/model/gmc/predictions'

ordered_keys = [
    'true_label',
    'filename',
    'classified_as',
    'vae_means_cos_r1',
    'vae_means_cos_r5',
    'vae_means_cos_r10',
    'vae_means_cos_r25',
    'vae_means_cos_r50',
    'vae_means_euc_r1',
    'vae_means_euc_r5',
    'vae_means_euc_r10',
    'vae_means_euc_r25',
    'vae_means_euc_r50',
    'vae_exemplars_nearest_cos_r1',
    'vae_exemplars_nearest_cos_r5',
    'vae_exemplars_nearest_cos_r10',
    'vae_exemplars_nearest_cos_r25',
    'vae_exemplars_nearest_cos_r50',
    'vae_exemplars_nearest_euc_r1',
    'vae_exemplars_nearest_euc_r5',
    'vae_exemplars_nearest_euc_r10',
    'vae_exemplars_nearest_euc_r25',
    'vae_exemplars_nearest_euc_r50',
    'vae_exemplars_random_cos_r1',
    'vae_exemplars_random_cos_r5',
    'vae_exemplars_random_cos_r10',
    'vae_exemplars_random_cos_r25',
    'vae_exemplars_random_cos_r50',
    'vae_exemplars_random_euc_r1',
    'vae_exemplars_random_euc_r5',
    'vae_exemplars_random_euc_r10',
    'vae_exemplars_random_euc_r25',
    'vae_exemplars_random_euc_r50',
    'classifier_r1',
    'classifier_r5',
    'classifier_r10',
    'classifier_r25',
    'classifier_r50',
    'classifier_p5',
    'classifier_p75',
    'classifier_p9'
]

n = len(words_to_evaluate)
for i, w1 in enumerate(words_to_evaluate):
    print('{0} of {1}'.format(i+1, n), end='\r')
    if os.path.exists(os.path.join(entailment_path, w1)):
        continue
    with open(os.path.join(entailment_path, w1), 'w+') as g:
        g.write('{0}\n'.format('\t'.join(ordered_keys)))
        for w2 in words_to_evaluate:
            if (w1, w2) not in wordnet_gmc_pairs:
                continue

            try:
                filename_dict = defaultdict(dict)
                for exemplar, path in vae_prediction_paths.items():
                    with open(os.path.join(path, w1 + '.json')) as f:
                        for line in f:
                            line = json.loads(line)
                            filename_dict[line['filename']][exemplar] = line
                with open(os.path.join(classifier_prediction_path, w1 + '.json')) as f:
                    for line in f:
                        line = json.loads(line)
                        filename_dict[line['filename']]['classifier'] = line
            except:
                print('Failed to find predictions for {0}'.format(w1))
                break

            for filename in filename_dict.keys():
                entry = {'true_label': w1, 'filename': filename, 'classified_as': w2}

                for vae_model in vae_prediction_paths.keys():
                    vae_data = filename_dict[filename][vae_model]
                    for r in [1, 5, 10, 25, 50]:
                        for method in ['cos', 'euc']:
                            metric = 'vae_' + vae_model + '_' + method + '_r' + str(r)
                            membership = str(int(w2 in vae_data['predictions_' + method][:r]))
                            entry[metric] = membership

                classifier_data = filename_dict[filename]['classifier']
                for r in [1, 5, 10, 25, 50]:
                    metric = 'classifier_r' + str(r)
                    membership = str(int(w2 in classifier_data['predictions'][:r]))
                    entry[metric] = membership
                for p, val in [('p5', .5), ('p75', .75), ('p9', .9)]:
                    metric = 'classifier_' + p
                    membership = str(int(classifier_data['p'] >= val))
                    entry[metric] = membership

                g.write('{0}\n'.format('\t'.join([entry[key] for key in ordered_keys])))
