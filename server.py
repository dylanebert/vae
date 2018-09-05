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
app = Flask(__name__)

class Properties:
    def __init__(self, data_path, save_path, latent_size):
        self.params = Params()
        self.train_generator = DataGenerator(os.path.join(data_path, 'train'), self.params)
        self.dev_generator = DataGenerator(os.path.join(data_path, 'dev'), self.params)
        self.test_generator = DataGenerator(os.path.join(data_path, 'test'), self.params)
        self.num_classes = self.train_generator.num_classes()

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
        save_path = os.path.join(request.args.get('savepath'), 'weights_best.h5')
        assert(os.path.exists(save_path))
    except:
        return 'error: invalid save path'
    try:
        latent_size = int(request.args.get('latentsize'))
        assert(latent_size > 0)
    except:
        return 'error: invalid latent size'
    try:
        properties = Properties(data_path, save_path, latent_size)
    except:
        return 'error: failed to create data generator'
    try:
        network = VAE(properties.params)
        network.vae.load_weights(save_path)
        initialized = True
        return 'success'
    except:
        return 'error: failed to build network'

@app.route('/classes')
def get_classes():
    return ['test1', 'test2']

if __name__ == '__main__':
    app.run()
