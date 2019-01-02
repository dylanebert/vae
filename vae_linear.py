from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras import metrics
from keras import optimizers
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

class VAE():
    def __init__(self):
        self.original_size = 2048
        self.input_shape = (self.original_size,)
        self.h1_size = 128
        self.h2_size = 64
        self.latent_size = 2

        x = Input(shape=self.input_shape, name='encoder_input')
        h1 = Dense(self.h1_size, activation='relu')(x)
        h2 = Dense(self.h2_size, activation='relu')(h1)
        z_mean = Dense(self.latent_size, name='z_mean')(h2)
        z_stddev = Dense(self.latent_size, name='z_stddev')(h2)

        def sampling(args):
            z_mean, z_stddev = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_size), mean=0., stddev=1.0)
            return z_mean + K.exp(z_stddev) * epsilon

        z = Lambda(sampling, output_shape=(self.latent_size,), name='z')([z_mean, z_stddev])

        self.encoder = Model(x, [z_mean, z_stddev, z], name='encoder')
        self.encoder.summary()

        z_in = Input(shape=(self.latent_size,), name='z_sampling')
        h1_dec = Dense(self.h2_size, activation='relu')(z_in)
        h2_dec = Dense(self.h1_size, activation='relu')(h1_dec)
        x_reconstr = Dense(self.original_size, activation='sigmoid', name='reconstruction')(h2_dec)

        self.decoder = Model(z_in, x_reconstr, name='decoder')
        self.decoder.summary()

        x_reconstr = self.decoder(self.encoder(x)[2])
        self.vae = Model(x, x_reconstr, name='vae')

        xent_loss = self.original_size * metrics.binary_crossentropy(K.flatten(x), K.flatten(x_reconstr))
        kl_loss = -0.5 * K.sum(1 + z_stddev - K.square(z_mean) - K.exp(z_stddev), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        self.vae.add_loss(vae_loss)

        optimizer = optimizers.Adam(lr=1e-3)
        self.vae.compile(optimizer=optimizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    model = VAE()
