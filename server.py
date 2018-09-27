import os
from flask import Flask, request
import pickle
import json
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
    if model is None:
        return 'No model has been loaded'
    else:
        return str(model)

@app.route('/classes')
def classes():
    if model is None:
        return 'No model has been loaded'
    else:
        return json.dumps(sorted(list(model.encodings.keys())))

if __name__ == '__main__':
    app.run()
