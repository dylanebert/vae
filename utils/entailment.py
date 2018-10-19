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
    'label',
    'filename',
    'entails',
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
    with open(os.path.join(entailment_path, w1), 'w+') as g:
        g.write('{0}\n'.format('\t'.join(ordered_keys)))
        for w2 in words_to_evaluate:
            if (w1, w2) not in wordnet_gmc_pairs and (w2, w1) not in wordnet_gmc_pairs:
                continue

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

            for filename in filename_dict.keys():
                entry = {'label': w1, 'filename': filename, 'entails': w2}

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

'''for w1 in words_to_evaluate:
    data = defaultdict(dict)

    #VAE recall
    for exemplar, path in vae_prediction_paths.items():
        filepath = os.path.join(path, w1 + '.json')
        with open(filepath, 'r') as f:
            for line in f:
                line = json.loads(line)
                for w2 in words_to_evaluate:
                    filename = line['filename']
                    for r in [1, 5, 10, 25, 50]:
                        for method in ['cos', 'euc']:
                            metric = 'vae_' + exemplar + '_' + method + '_r' + str(r)
                            membership = str(int(w2 in line['predictions_' + method][:r]))
                            data[filename][w2][metric] = membership

    #Classifier recall and pVals
    with open(os.path.join(classifier_prediction_path, word + '.json'), 'r') as f:
        for line in f:
            line = json.loads(line)
            filename = line['filename']
            data[filename]['filename'] = filename
            for r in [1, 5, 10, 25, 50]:
                metric = 'classifier_r' + str(r)
                membership = str(int(word in line['predictions'][:r]))
                data[filename][metric] = membership
            for p, val in [('p5', .5), ('p75', .75), ('p9', .9)]:
                metric = 'classifier_' + p
                membership = str(int(line['p'] >= val))
                data[filename][metric] = membership

    #Write to tab delimited file
    with open(os.path.join(entailment_path, word), 'w+') as f:
        f.write('{0}\n'.format('\t'.join(ordered_keys)))
        for filename, metrics in data.items():
            f.write('{0}\n'.format('\t'.join([metrics[key] for key in ordered_keys])))
    break'''

'''dir = 'model/gmc/predictions_test'
dirs = os.listdir(dir)
n = len(dirs)
for i, filename in enumerate(dirs):
    print('{0} of {1}'.format(i+1, n), end='\r')
    filepath = os.path.join(dir, filename)
    classifier_path = os.path.join('/home/dylan/Documents/inception/model/gmc/predictions', filename)
    entailment_path = os.path.join('entailment_test', filename.split('.')[0])
    with open(filepath, 'r') as f:
        #with open(classifier_path, 'r') as h:
            with open(entailment_path, 'w+') as g:
                g.write('FILENAME\tWORD\tENTAILS\tR1_COS\tR5_COS\tR10_COS\tR25_COS\tR50_COS\tR1_EUC\tR5_EUC\tR10_EUC\tR25_EUC\tR50_EUC\tR1_C\tR5_C\tR10_C\tR25_C\tR50_C\tP>.5\tP>.75\tP>.9\n')
                filename_dict_vae = {}
                filename_dict_classifier = {}
                for line in f:
                    entry = json.loads(line)
                    filename_dict_vae[entry['filename']] = entry
                for line in h:
                    entry = json.loads(line)
                    filename_dict_classifier[entry['filename']] = entry
                for filename, vae_entry in filename_dict_vae.items():
                    classifier_entry = filename_dict_classifier[filename]
                    for word in gmc_words:
                        pair = (entry['label'], word)
                        if pair not in wordnet_gmc_pairs and (pair[1], pair[0]) not in wordnet_gmc_pairs:
                            continue
                        r1_cos = word in vae_entry['predictions_cos'][:1]
                        r5_cos = word in vae_entry['predictions_cos'][:5]
                        r10_cos = word in vae_entry['predictions_cos'][:10]
                        r25_cos = word in vae_entry['predictions_cos'][:25]
                        r50_cos = word in vae_entry['predictions_cos'][:50]
                        r1_euc = word in vae_entry['predictions_euc'][:1]
                        r5_euc = word in vae_entry['predictions_euc'][:5]
                        r10_euc = word in vae_entry['predictions_euc'][:10]
                        r25_euc = word in vae_entry['predictions_euc'][:25]
                        r50_euc = word in vae_entry['predictions_euc'][:50]
                        r1 = word in classifier_entry['predictions'][:1]
                        r5 = word in classifier_entry['predictions'][:5]
                        r10 = word in classifier_entry['predictions'][:10]
                        r25 = word in classifier_entry['predictions'][:25]
                        r50 = word in classifier_entry['predictions'][:50]
                        p5 = float(classifier_entry['p']) >= .5
                        p75 = float(classifier_entry['p']) >= .75
                        p9 = float(classifier_entry['p']) >= .9
                        g.write('%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n' % (filename, entry['label'], word, r1_cos, r5_cos, r10_cos, r25_cos, r50_cos, r1_euc, r5_euc, r10_euc, r25_euc, r50_euc, r1, r5, r10, r25, r50, p5, p75, p9))'''
