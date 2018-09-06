import tensorflow as tf
import keras
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras import metrics
from callbacks import Histories
import numpy as np
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from params import Params
from flask import Flask, request
from data_generator import DataGenerator
from vae import VAE
import json
import pickle
import base64
app = Flask(__name__)

class Properties:
    def __init__(self, data_path, means_path, latent_size):
        self.params = Params()
        self.latent_size = latent_size
        self.train_generator = DataGenerator(os.path.join(data_path, 'train'), self.params)
        self.dev_generator = DataGenerator(os.path.join(data_path, 'dev'), self.params)
        self.test_generator = DataGenerator(os.path.join(data_path, 'test'), self.params)
        self.num_classes = self.train_generator.num_classes()
        self.indices = self.train_generator.get_indices()
        self.class_means = pickle.load(open(means_path, 'rb'))

properties = None
network = None
initialized = False

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/init')
def init():
    global properties, network, initialized
    if initialized == True:
        return 'success'
    try:
        data_path = request.args.get('datapath')
        assert(os.path.exists(data_path))
    except:
        return 'error: invalid data path'
    try:
        weights_path = os.path.join(request.args.get('savepath'), 'weights_best.h5')
        means_path = os.path.join(request.args.get('savepath'), 'means.p')
        assert(os.path.exists(weights_path))
        assert(os.path.exists(means_path))
    except:
        return 'error: invalid save path'
    try:
        latent_size = int(request.args.get('latentsize'))
        assert(latent_size > 0)
    except:
        return 'error: invalid latent size'
    try:
        properties = Properties(data_path, means_path, latent_size)
    except:
        return 'error: failed to create data generator'
    try:
        network = VAE(properties.params)
        network.vae.load_weights(weights_path)
        properties.decoded = network.generator.predict(properties.class_means, verbose=1)
        initialized = True
        return 'success'
    except:
        return 'error: failed to build network'

@app.route('/classes')
def get_classes():
    return json.dumps(list(properties.train_generator.class_names().values()))

@app.route('/get')
def get():
    class_name = request.args.get('class')
    class_index = properties.indices[class_name]
    img_matrix = properties.decoded[class_index]
    img = base64.b64encode(img_matrix)
    return img

if __name__ == '__main__':
    app.run()
