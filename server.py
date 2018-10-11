import os
from flask import Flask, request, send_file
import pickle
import json
import base64
import numpy as np
from config import Config
app = Flask(__name__)

class Model:
    def __init__(self, config):
        self.config = config
        with open(config.means_reduced_path, 'rb') as f:
            self.means_reduced = pickle.load(f)
        with open(config.encodings_reduced_path, 'rb') as f:
            self.encodings_reduced = pickle.load(f)

    def __str__(self):
        return str(self.config)

#Initialization
global model
config_path = 'model/gmc/config.json'
config = Config()
with open(config_path, 'r') as f:
    config.__dict__ = json.load(f)
model = Model(config)
print('Finished initialization')

inception_predictions_path = '/home/dylan/Documents/inception/model/gmc/predictions'
vae_predictions_path = 'model/gmc/predictions'

def get_r_sets(path, target, method):
    r_vals = [[] for i in range(5)]
    r_thresholds = [1, 5, 10, 25, 50]
    with open(path, 'r') as f:
        for line in f:
            parsed = json.loads(line)
            preds = parsed[method]
            done = False
            for r in r_thresholds:
                if target in preds[:r]:
                    for i, threshold in enumerate(r_thresholds):
                        if threshold >= r:
                            r_vals[i].append(1)
                        else:
                            r_vals[i].append(0)
                    done = True
                    break
            if not done:
                for i, threshold in enumerate(r_thresholds):
                    r_vals[i].append(0)
    return r_vals

def get_recall(path, label, method):
    r_sets = get_r_sets(path, label, method)
    r_means = np.mean(r_sets, axis=1)
    return r_means

def get_inception_probs(path, label):
    inception_predictions = []
    with open(path, 'r') as f:
        for line in f:
            inception_predictions.append(json.loads(line))

    p5 = []
    p75 = []
    p9 = []

    for line in inception_predictions:
        p = float(line['p'])
        if p >= .9:
            p5.append(1)
            p75.append(1)
            p9.append(1)
        elif p >= .75:
            p5.append(1)
            p75.append(1)
            p9.append(0)
        elif p>= .5:
            p5.append(1)
            p75.append(0)
            p9.append(0)
        else:
            p5.append(0)
            p75.append(0)
            p9.append(0)

    return np.mean(p5), np.mean(p75), np.mean(p9)

@app.route('/classes')
def classes():
    return json.dumps(sorted(list(model.encodings_reduced.keys())))

@app.route('/data')
def data():
    data = {}
    label = request.args.get('label')
    with open(os.path.join(model.config.image_path, 'means/{0}.jpg'.format(label)), 'rb') as f:
        data['img_mean'] = base64.b64encode(f.read()).decode('utf-8')
    with open(os.path.join(model.config.image_path, 'nearest/{0}.jpg'.format(label)), 'rb') as f:
        data['img_nearest'] = base64.b64encode(f.read()).decode('utf-8')
    with open(os.path.join(model.config.image_path, 'random/{0}.jpg'.format(label)), 'rb') as f:
        data['img_random'] = base64.b64encode(f.read()).decode('utf-8')
    mean_reduced = model.means_reduced[label].tolist()
    encodings_reduced = model.encodings_reduced[label].tolist()
    mean_reduced = {'x': mean_reduced[0], 'y': mean_reduced[1]}
    encodings_reduced = [{'x': x[0], 'y': x[1]} for x in encodings_reduced]
    r1_cos, r5_cos, r10_cos, r25_cos, r50_cos = get_recall(os.path.join(vae_predictions_path, label + '.json'), label, 'predictions_cos')
    r1_euc, r5_euc, r10_euc, r25_euc, r50_euc = get_recall(os.path.join(vae_predictions_path, label + '.json'), label, 'predictions_euc')
    r1_c, r5_c, r10_c, r25_c, r50_c = get_recall(os.path.join(inception_predictions_path, label + '.json'), label, 'predictions')
    p5, p75, p9 = get_inception_probs(os.path.join(inception_predictions_path, label + '.json'), label)
    data['mean'] = mean_reduced
    data['encodings'] = encodings_reduced
    data['r1-an'] = r1_c
    data['r5-an'] = r5_c
    data['r10-an'] = r10_c
    data['r25-an'] = r25_c
    data['r50-an'] = r50_c
    data['r1-cos'] = r1_cos
    data['r5-cos'] = r5_cos
    data['r10-cos'] = r10_cos
    data['r25-cos'] = r25_cos
    data['r50-cos'] = r50_cos
    data['r1-euc'] = r1_euc
    data['r5-euc'] = r5_euc
    data['r10-euc'] = r10_euc
    data['r25-euc'] = r25_euc
    data['r50-euc'] = r50_euc
    data['p5'] = p5
    data['p75'] = p75
    data['p9'] = p9
    return json.dumps(data)

@app.route('/entailment')
def entailment():
    data = {}
    c1 = request.args.get('c1')
    c2 = request.args.get('c2')
    r1_cos, r5_cos, r10_cos, r25_cos, r50_cos = get_recall(os.path.join(vae_predictions_path, c1 + '.json'), c2, 'predictions_cos')
    r1_euc, r5_euc, r10_euc, r25_euc, r50_euc = get_recall(os.path.join(vae_predictions_path, c1 + '.json'), c2, 'predictions_euc')
    r1_c, r5_c, r10_c, r25_c, r50_c = get_recall(os.path.join(inception_predictions_path, c1 + '.json'), c2, 'predictions')
    p5, p75, p9 = get_inception_probs(os.path.join(inception_predictions_path, c1 + '.json'), c2)
    data['r1-an'] = r1_c
    data['r5-an'] = r5_c
    data['r10-an'] = r10_c
    data['r25-an'] = r25_c
    data['r50-an'] = r50_c
    data['r1-cos'] = r1_cos
    data['r5-cos'] = r5_cos
    data['r10-cos'] = r10_cos
    data['r25-cos'] = r25_cos
    data['r50-cos'] = r50_cos
    data['r1-euc'] = r1_euc
    data['r5-euc'] = r5_euc
    data['r10-euc'] = r10_euc
    data['r25-euc'] = r25_euc
    data['r50-euc'] = r50_euc
    data['p5'] = p5
    data['p75'] = p75
    data['p9'] = p9
    return json.dumps(data)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
