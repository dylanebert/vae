import json
import os

with open('gmc_words', 'r') as f:
    gmc_words = f.read().splitlines()
with open('wordnet_gmc_pairs', 'r') as f:
    wordnet_gmc_pairs = [tuple(line.rstrip().split(' ')) for line in f]

dir = 'model/gmc/predictions_test'
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
                        g.write('%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n' % (filename, entry['label'], word, r1_cos, r5_cos, r10_cos, r25_cos, r50_cos, r1_euc, r5_euc, r10_euc, r25_euc, r50_euc, r1, r5, r10, r25, r50, p5, p75, p9))
