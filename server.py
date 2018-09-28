import os
from flask import Flask, request, send_file
import pickle
import json
import base64
from config import Config
app = Flask(__name__)

class Model:
    def __init__(self, config):
        self.config = config
        with open(config.means_path, 'rb') as f:
            self.means = pickle.load(f)
        with open(config.encodings_path, 'rb') as f:
            self.encodings = pickle.load(f)

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

@app.route('/image')
def image():
    label = request.args.get('label')
    path = os.path.join(model.config.image_path, label + '.jpg')
    if not os.path.exists(path):
        return 'Image not found'
    else:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

if __name__ == '__main__':
    app.run()
