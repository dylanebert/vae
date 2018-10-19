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
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class VAE:
    def __init__(self, config):
        self.config = config
        self.train_generator = None
        self.dev_generator = None
        self.test_generator = None
        self.data_loaded = False
        self.encodings = {}
        self.class_means = {}

        self.weights_path = os.path.join(config.model_path, 'weights_best.p')
        self.overfit_path = os.path.join(config.model_path, 'weights_overfit.p')
        self.logs_path = os.path.join(config.model_path, 'logs')
        self.images_path = os.path.join(config.model_path, 'images')
        self.encodings_path = os.path.join(config.model_path, 'encodings', 'all.p')
        self.means_path = os.path.join(config.model_path, 'encodings', 'means.p')
        self.train_path = os.path.join(config.data_path, 'train')
        self.dev_path = os.path.join(config.data_path, 'dev')
        self.test_path = os.path.join(config.data_path, 'test')

        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        if not os.path.exists(os.path.join(config.model_path, 'encodings')):
            os.makedirs(os.path.join(config.model_path, 'encodings'))
        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)

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
            self.vae.load_weights(config.weights_path)
            print('Loaded weights')
        except:
            print('Couldn\'t find/load weights')

    def build_generators(self):
        self.train_generator = DataGenerator(self.train_path, self.config.image_size, self.config.batch_size)
        self.dev_generator = DataGenerator(self.dev_path, self.config.image_size, self.config.batch_size)
        self.test_generator = DataGenerator(self.test_path, self.config.image_size, self.config.batch_size)

    def train(self, max_epochs=1000, overfit=False):
        if not self.data_loaded:
            self.build_generators()
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logs_path)
        checkpoint_callback = keras.callbacks.ModelCheckpoint(self.weights_path, save_best_only=True, verbose=1)
        callbacks = [tensorboard_callback, checkpoint_callback]
        if not overfit:
            earlystopping_callback = keras.callbacks.EarlyStopping(verbose=1, patience=2)
            callbacks.append(earlystopping_callback)
        self.vae.fit_generator(generator=self.train_generator, validation_data=self.dev_generator, epochs=max_epochs, callbacks=callbacks)
        self.vae.save_weights(self.overfit_path)

    def compute_encodings(self):
        if not self.data_loaded:
            self.build_generators()
        z = self.encoder.predict_generator(self.train_generator, verbose=1)
        class_index_dict = self.train_generator.generator.class_indices
        index_class_dict = {k: v for v, k in class_index_dict.items()}
        filenames = self.train_generator.generator.filenames
        num_classes = len(index_class_dict)
        n = len(self.train_generator)
        print('Computing encodings')
        for i in range(n):
            print('{0} of {1}'.format(i+1, n), end='\r')
            _, y = self.train_generator.generator[i]
            for j, class_index in enumerate(y):
                class_name = index_class_dict[class_index]
                if class_name not in self.encodings:
                    self.encodings[class_name] = {'encodings': [], 'filenames': []}
                self.encodings[class_name]['encodings'].append(z[self.config.batch_size * i + j].tolist())
                self.encodings[class_name]['filenames'].append(filenames[self.config.batch_size * i + j])
        with open(self.config.encodings_path, 'wb+') as f:
            pickle.dump(self.encodings, f)

    def compute_means(self):
        try:
            with open(self.encodings_path, 'rb') as f:
                self.encodings = pickle.load(f)
            print('Loaded encodings')
        except:
            self.compute_encodings()
        print('Computing class means')
        self.class_means = {}
        for label, entry in self.encodings.items():
            encodings = entry['encodings']
            self.class_means[label] = np.mean(encodings, axis=0).tolist()
        with open(self.means_path, 'wb+') as f:
            pickle.dump(self.class_means, f)

    def reconstruct_means(self):
        try:
            with open(self.config.means_path, 'rb') as f:
                self.class_means = pickle.load(f)
            print('Loaded means')
        except:
            self.compute_means()
        print('Decoding class means')
        class_names = list(self.class_means.keys())
        class_means = np.array(list(self.class_means.values()))
        decoded = self.generator.predict(class_means)
        for i, img in enumerate(decoded):
            class_name = class_names[i]
            image_path = os.path.join(self.config.image_path, class_name + '.jpg')
            imsave(image_path, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='data directory, containg train/dev/test folders', type=str, required=True)
    parser.add_argument('--model_path', help='directory in which to store', type=str, required=True)
    parser.add_argument('--image_size', help='square height/width at which to process input', type=int, default=64)
    parser.add_argument('--filters', help='number of convolution filters', type=int, default=64)
    parser.add_argument('--latent_size', help='dimension of latent space', type=int, default=300)
    parser.add_argument('--batch_size', help='number of images per batch', type=int, default=100)
    parser.add_argument('--learning_rate', help='model learning rate', type=float, default=.001)
    parser.add_argument('--train', help='train for given max epochs', type=int, default=0)
    parser.add_argument('--compute_encodings', help='compute and store train encodings', action='store_true')
    parser.add_argument('--compute_means', help='compute and store mean encoding of each word', action='store_true')
    parser.add_argument('--reconstruct_means', help='decode and store image corresponding to each mean', action='store_true')
    parser.add_argument('-a', '--all', help='shorthand to perform all training procedures', action='store_true')
    args = parser.parse_args()
    if not args.train and not args.compute_encodings and not args.compute_means and not args.reconstruct_means and not args.all:
        sys.exit('Error: at least one command is required (train/compute_encodings/compute_means/reconstruct_means/all)')

    if args.data_path is not None and not os.path.exists(args.data_path):
        sys.exit('Error: data path {0} not found'.format(args.data_path))
    config = Config(args.data_path, args.model_path, args.image_size, args.filters, args.latent_size, args.batch_size, args.learning_rate)
    if args.model_path is not None and os.path.exists(os.path.join(args.model_path, 'config.json')):
        config.load(os.path.join(args.model_path, 'config.json'))

    vae = VAE(config)
    if args.train is not 0:
        vae.train(args.train)
    elif args.all:
        vae.train()
    if args.compute_encodings or args.all:
        vae.compute_encodings()
    if args.compute_means or args.all:
        vae.compute_means()
    if args.reconstruct_means or args.all:
        vae.reconstruct_means()
    config.save()
