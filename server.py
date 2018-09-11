import os
import pickle
from config import Config
from vae import VAE
from flask import Flask, request
app = Flask(__name__)

config = None

@app.route('/new')
def new():
    global config
    config = Config()
    try:
        pickle.dump(config, open(config.save_path, 'wb+'))
    except:
        return 'failed to save config'
    return 'done'

@app.route('/load')
def load():
    global config
    try:
        path = request.args.get('path')
    except:
        return 'missing argument: path'
    if not os.path.exists(path):
        return 'invalid path'
    try:
        config = pickle.load(open(path, 'rb'))
    except:
        return 'failed to load config file'
    return 'done'

@app.route('/train')
def train():
    vae = VAE(config)
    vae.train()
    return 'done'

if __name__ == '__main__':
    app.run()
