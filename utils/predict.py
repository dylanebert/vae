means = self.class_means
encodings = self.test_encodings
k = 0
n = len(encodings)
for label, entry in encodings.items():
    encs = entry['encodings']
    filenames = entry['filenames']
    path = os.path.join(self.config.predictions_path, label + '.json')
    if os.path.exists(path):
        os.remove(path)
    k += 1
    print('{0} of {1}'.format(k, n), end='\r')
    for i, enc in enumerate(encs):
        try:
            nearest_cos = list(dict(sorted(means.items(), key=lambda x: cosine(enc, x[1]))[:100]).keys())
            nearest_euc = list(dict(sorted(means.items(), key=lambda x: euclidean(enc, x[1]))[:100]).keys())
            line = json.dumps({'label': label, 'filename': filenames[i], 'cos': cosine(enc, means[label]), 'euc': euclidean(enc, means[label]), 'predictions_cos': nearest_cos, 'predictions_euc': nearest_euc})
            with open(path, 'a+') as f:
                f.write('{0}\n'.format(line))
        except:
            print('Failed on {0}'.format(label))
