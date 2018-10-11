import sys
sys.path.insert(0, '/home/dylan/Documents/vae')
from vae import VAE
from config import Config
from data_generator import DataGenerator
import numpy as np
from scipy.misc import imsave
import os

config = Config()
config.load('model/gmc/config.json')
vae = VAE(config)

datagen = DataGenerator('/data/gmc_exemplars', config.image_size, config.batch_size)
filenames = datagen.generator.filenames
reconstr = vae.vae.predict_generator(datagen, verbose=1)
for i, img in enumerate(reconstr):
    filename = filenames[i]
    filepath = os.path.join('model/gmc/images', filename)
    imsave(filepath, img)
