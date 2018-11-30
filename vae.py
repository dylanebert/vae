import tensorflow as tf
import keras
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras import metrics
from data_generator import DataGenerator
from scipy.misc import imsave
import numpy as np
import os
import pickle
import json
from sklearn.decomposition import PCA
import argparse
import sys
from config import Config
from tqdm import tqdm
import h5py
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class VAE:
    def __init__(self, config):
        self.config = config

        self.weights_path = os.path.join(config.model_path, 'weights_best.p')
        self.overfit_path = os.path.join(config.model_path, 'weights_overfit.p')
        self.logs_path = os.path.join(config.model_path, 'logs')
        self.images_path = os.path.join(config.model_path, 'images')
        self.train_path = os.path.join(config.data_path, 'train')
        self.dev_path = os.path.join(config.data_path, 'dev')
        self.test_path = os.path.join(config.data_path, 'test')

        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        image_size = config.image_size
        filters = config.filters
        latent_size = config.latent_size
        batch_size = config.batch_size
        learning_rate = config.learning_rate

        x = Input(shape=(image_size, image_size, 3))

        conv1 = Conv2D(3, kernel_size=(2, 2), padding='same', activation='relu')(x)
        conv2 = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu', strides=(2, 2))(conv1)
        conv3 = Conv2D(filters, kernel_size=3, padding='same', activation='relu', strides=1)(conv2)
        conv4 = Conv2D(filters, kernel_size=3, padding='same', activation='relu', strides=1)(conv3)
        flat = Flatten()(conv4)

        z_mean = Dense(latent_size)(flat)
        z_stddev = Dense(latent_size)(flat)

        def sampling(args):
            z_mean, z_stddev = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_size), mean=0., stddev=1.0)
            return z_mean + K.exp(z_stddev) * epsilon

        z = Lambda(sampling, output_shape=(latent_size,))([z_mean, z_stddev])

        decoder_upsample = Dense(filters * (image_size // 2) * (image_size // 2), activation='relu')

        output_shape = (batch_size, image_size // 2, image_size // 2, filters)

        decoder_reshape = Reshape(output_shape[1:])
        decoder_deconv1 = Conv2DTranspose(filters, kernel_size=3, padding='same', strides=1, activation='relu')
        decoder_deconv2 = Conv2DTranspose(filters, kernel_size=3, padding='same', strides=1, activation='relu')

        output_shape = (batch_size, filters, image_size + 1, image_size + 1)

        decoder_deconv3_upsamp = Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')
        decoder_reconstr = Conv2D(3, kernel_size=2, padding='valid', activation='sigmoid')

        up_decoded = decoder_upsample(z)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv1_decoded = decoder_deconv1(reshape_decoded)
        deconv2_decoded = decoder_deconv2(deconv1_decoded)
        x_decoded_relu = decoder_deconv3_upsamp(deconv2_decoded)
        x_reconstr = decoder_reconstr(x_decoded_relu)

        self.vae = Model(x, x_reconstr)

        xent_loss = image_size * image_size * metrics.binary_crossentropy(K.flatten(x), K.flatten(x_reconstr))
        kl_loss = -0.5 * K.sum(1 + z_stddev - K.square(z_mean) - K.exp(z_stddev), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        self.vae.add_loss(vae_loss)

        optimizer = optimizers.Adam(lr=learning_rate)
        self.vae.compile(optimizer=optimizer)
        self.vae.summary()

        self.encoder = Model(x, z_mean)

        decoder_input = Input(shape=(latent_size,))
        _up_decoded = decoder_upsample(decoder_input)
        _reshape_decoded = decoder_reshape(_up_decoded)
        _deconv1_decoded = decoder_deconv1(_reshape_decoded)
        _deconv2_decoded = decoder_deconv2(_deconv1_decoded)
        _x_decoded_relu = decoder_deconv3_upsamp(_deconv2_decoded)
        _x_reconstr = decoder_reconstr(_x_decoded_relu)
        self.generator = Model(decoder_input, _x_reconstr)

        try:
            self.vae.load_weights(self.weights_path)
            print('Loaded weights')
        except:
            print('Couldn\'t find/load weights')

    def train(self, max_epochs=1000, overfit=False):
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logs_path)
        checkpoint_callback = keras.callbacks.ModelCheckpoint(self.weights_path, save_best_only=True, verbose=1)
        callbacks = [tensorboard_callback, checkpoint_callback]
        if not overfit:
            earlystopping_callback = keras.callbacks.EarlyStopping(verbose=1, patience=2)
            callbacks.append(earlystopping_callback)
        train_generator = DataGenerator(self.train_path, self.config.image_size, self.config.batch_size, train=True)
        dev_generator = DataGenerator(self.dev_path, self.config.image_size, self.config.batch_size)
        self.vae.fit_generator(generator=train_generator, validation_data=dev_generator, epochs=max_epochs, callbacks=callbacks)
        self.vae.save_weights(self.overfit_path)

    def encode(self):
        train_generator = DataGenerator(self.train_path, self.config.image_size, self.config.batch_size)
        z = self.encoder.predict_generator(train_generator, verbose=1)

        with h5py.File(os.path.join(self.model_path, 'encodings.hdf5'), 'w') as f:
            f.create_dataset('encodings', data=z)
            f.create_dataset('filenames', data=np.array(train_generator.generator.filenames, dtype='S'))

    def reconstruct(self, encodings_path, images_path):
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        with h5py.File(encodings_path, 'r') as f:
            encodings = f['encodings']
            filenames = [filename.decode('utf-8') for filename in f['filenames']]
            decoded = self.generator.predict(encodings, verbose=1)
        for i in tqdm(range(len(decoded))):
            img = decoded[i]
            dir, filename = os.path.split(filenames[i])
            dirpath = os.path.join(images_path, dir)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            imsave(os.path.join(dirpath, filename), img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='directory in which to store', type=str, required=True)
    parser.add_argument('--data_path', help='data directory, containg train/dev/test folders', type=str, default='')
    parser.add_argument('--image_size', help='square height/width at which to process input', type=int, default=64)
    parser.add_argument('--filters', help='number of convolution filters', type=int, default=64)
    parser.add_argument('--latent_size', help='dimension of latent space', type=int, default=300)
    parser.add_argument('--batch_size', help='number of images per batch', type=int, default=100)
    parser.add_argument('--learning_rate', help='model learning rate', type=float, default=.001)
    parser.add_argument('--train', help='train for given max epochs', type=int, default=0)
    parser.add_argument('--encode', help='encode training data and store vectors', action='store_true')
    parser.add_argument('--reconstruct', help='reconstruct vectors at given filepath (relative to model) and save images', type=str, default='')
    parser.add_argument('--image_path', help='override default save path (relative to model) for reconstructions', type=str, default='images')
    args = parser.parse_args()

    config = Config(args.data_path, args.model_path, args.image_size, args.filters, args.latent_size, args.batch_size, args.learning_rate)
    if os.path.exists(os.path.join(args.model_path, 'config.json')):
        config.load(os.path.join(args.model_path, 'config.json'))
    if not os.path.exists(config.data_path):
        sys.exit('Error: data path {0} not found'.format(config.data_path))

    vae = VAE(config)
    if args.train is not 0:
        vae.train(args.train)
    if args.encode:
        vae.encode()
    if args.reconstruct is not '':
        vae.reconstruct(os.path.join(config.model_path, args.reconstruct), os.path.join(config.model_path, args.image_path))
    config.save()
