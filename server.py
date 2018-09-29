import os
from flask import Flask, request, send_file
import pickle
import json
import base64
import numpy as np
from sklearn.decomposition import PCA
from config import Config
app = Flask(__name__)

class Model:
    def __init__(self, config):
        self.config = config
        with open(config.means_path, 'rb') as f:
            self.means = pickle.load(f)
        with open(config.encodings_path, 'rb') as f:
            self.encodings = pickle.load(f)
        labels = list(self.means.keys())
        mean_vals = np.array(list(self.means.values()))
        pca = PCA(n_components=2)
        means_reduced = pca.fit_transform(mean_vals)
        encodings_reduced = {}
        for label in labels:
            encodings_reduced[label] = pca.transform(np.array(list(self.encodings[label])))
        self.means_reduced = dict(zip(labels, means_reduced))
        self.encodings_reduced = encodings_reduced

    def __str__(self):
        return str(self.config)

model = None

@app.route('/is-loaded')
def loaded():
    if model is None:
        return '0'
    else:
        return '1'

@app.route('/load')
def load():
    config_path = request.args.get('path')
    config = Config()
    with open(config_path, 'r') as f:
        config.__dict__ = json.load(f)
    global model
    model = Model(config)
    return '1'

@app.route('/test')
def test():
    return str(model)

@app.route('/classes')
def classes():
    return json.dumps(sorted(list(model.encodings.keys()))[:100])

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
    data['mean'] = mean_reduced
    data['encodings'] = encodings_reduced
    return json.dumps(data)

if __name__ == '__main__':
    app.run()
