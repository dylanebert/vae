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

def get_recall(label, path):
    inception_predictions = []
    with open(os.path.join(path, label + '.json'), 'r') as f:
        for line in f:
            inception_predictions.append(json.loads(line))

    r1 = []
    r5 = []
    r10 = []
    r25 = []
    r50 = []

    for line in inception_predictions:
        preds = line['predictions']
        if label in preds[:1]:
            r1.append(1)
            r5.append(1)
            r10.append(1)
            r25.append(1)
            r50.append(1)
        elif label in preds[:5]:
            r1.append(0)
            r5.append(1)
            r10.append(1)
            r25.append(1)
            r50.append(1)
        elif label in preds[:10]:
            r1.append(0)
            r5.append(0)
            r10.append(1)
            r25.append(1)
            r50.append(1)
        elif label in preds[:25]:
            r1.append(0)
            r5.append(0)
            r10.append(0)
            r25.append(1)
            r50.append(1)
        elif label in preds[:50]:
            r1.append(0)
            r5.append(0)
            r10.append(0)
            r25.append(0)
            r50.append(1)
        else:
            r1.append(0)
            r5.append(0)
            r10.append(0)
            r25.append(0)
            r50.append(0)

    return np.mean(r1), np.mean(r5), np.mean(r10), np.mean(r25), np.mean(r50)

def get_inception_probs(label, path):
    inception_predictions = []
    with open(os.path.join(path, label + '.json'), 'r') as f:
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
    return json.dumps(sorted(list(model.encodings.keys())))

@app.route('/data')
def data():
    data = {}
    label = request.args.get('label')
    path = os.path.join(model.config.image_path, label + '.jpg')
    if not os.path.exists(path):
        return 'Image not found'
    with open(path, 'rb') as f:
        data['img'] = base64.b64encode(f.read()).decode('utf-8')
    mean_reduced = model.means_reduced[label].tolist()
    encodings_reduced = model.encodings_reduced[label].tolist()
    mean_reduced = {'x': mean_reduced[0], 'y': mean_reduced[1]}
    encodings_reduced = [{'x': x[0], 'y': x[1]} for x in encodings_reduced]
    r1, r5, r10, r25, r50 = get_recall(label, 'inception_predictions')
    s1, s5, s10, s25, s50 = get_recall(label, 'model/gmc/predictions_cos')
    t1, t5, t10, t25, t50 = get_recall(label, 'model/gmc/predictions_euc')
    p5, p75, p9 = get_inception_probs(label, 'inception_predictions')
    data['mean'] = mean_reduced
    data['encodings'] = encodings_reduced
    data['r1-an'] = r1
    data['r5-an'] = r5
    data['r10-an'] = r10
    data['r25-an'] = r25
    data['r50-an'] = r50
    data['r1-cos'] = s1
    data['r5-cos'] = s5
    data['r10-cos'] = s10
    data['r25-cos'] = s25
    data['r50-cos'] = s50
    data['r1-euc'] = t1
    data['r5-euc'] = t5
    data['r10-euc'] = t10
    data['r25-euc'] = t25
    data['r50-euc'] = t50
    data['p5'] = p5
    data['p75'] = p75
    data['p9'] = p9
    return json.dumps(data)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
