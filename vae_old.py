import tensorflow as tf
import keras
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.models import model_from_json
from keras import optimizers
from keras import backend as K
from keras import metrics
from callbacks import Histories
import numpy as np
import json
import argparse
import os
import sys
import pickle
from scipy.stats import norm, multivariate_normal
from data_generator import DataGenerator
from sklearn.metrics import confusion_matrix
from params import Params
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class VAE:
    def __init__(self, params):
        image_size = params.image_size
        num_channels = params.num_channels
        filters = params.filters
        latent_size = params.latent_size
        batch_size = params.batch_size
        learning_rate = params.learning_rate

        input_shape = [image_size, image_size, num_channels]
        x = Input(shape=input_shape)

        #encoder
        conv1 = Conv2D(num_channels, kernel_size=(2, 2), padding='same', activation='relu')(x)
        conv2 = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu', strides=(2, 2))(conv1)
        conv3 = Conv2D(filters, kernel_size=3, padding='same', activation='relu', strides=1)(conv2)
        conv4 = Conv2D(filters, kernel_size=3, padding='same', activation='relu', strides=1)(conv3)
        flat = Flatten()(conv4)

        #latent space Z (mean and std)
        z_mean = Dense(latent_size)(flat)
        z_stddev = Dense(latent_size)(flat)

        #random sampling from Z
        def sampling(args):
            z_mean, z_stddev = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_size), mean=0., stddev=1.0)
            return z_mean + K.exp(z_stddev) * epsilon

        z = Lambda(sampling, output_shape=(latent_size,))([z_mean, z_stddev])

        #decoder
        decoder_upsample = Dense(filters * (image_size // 2) * (image_size // 2), activation='relu')

        output_shape = (batch_size, image_size // 2, image_size // 2, filters)

        decoder_reshape = Reshape(output_shape[1:])
        decoder_deconv1 = Conv2DTranspose(filters, kernel_size=3, padding='same', strides=1, activation='relu')
        decoder_deconv2 = Conv2DTranspose(filters, kernel_size=3, padding='same', strides=1, activation='relu')

        output_shape = (batch_size, filters, image_size + 1, image_size + 1)

        decoder_deconv3_upsamp = Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')
        decoder_mean_squash = Conv2D(num_channels, kernel_size=2, padding='valid', activation='sigmoid')

        up_decoded = decoder_upsample(z)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv1_decoded = decoder_deconv1(reshape_decoded)
        deconv2_decoded = decoder_deconv2(deconv1_decoded)
        x_decoded_relu = decoder_deconv3_upsamp(deconv2_decoded)
        x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

        #Model for full vae
        self.vae = Model(x, x_decoded_mean_squash)

        #Loss function
        xent_loss = image_size * image_size * metrics.binary_crossentropy(K.flatten(x), K.flatten(x_decoded_mean_squash))
        kl_loss = -0.5 * K.sum(1 + z_stddev - K.square(z_mean) - K.exp(z_stddev), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        self.vae.add_loss(vae_loss)

        optimizer = optimizers.Adam(lr=learning_rate)
        self.vae.compile(optimizer=optimizer)
        self.vae.summary()

        #model to project input onto latent space
        self.encoder = Model(x, z_mean)

        #model for generating image from latent vector
        decoder_input = Input(shape=(latent_size,))
        _up_decoded = decoder_upsample(decoder_input)
        _reshape_decoded = decoder_reshape(_up_decoded)
        _deconv1_decoded = decoder_deconv1(_reshape_decoded)
        _deconv2_decoded = decoder_deconv2(_deconv1_decoded)
        _x_decoded_relu = decoder_deconv3_upsamp(_deconv2_decoded)
        _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
        self.generator = Model(decoder_input, _x_decoded_mean_squash)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='override directory for data', type=str, default='')
    parser.add_argument('--save_path', help='override path to save files', type=str, default='')
    parser.add_argument('--nz', help='override latent dimension hyperparameter', type=int, default=0)
    parser.add_argument('--early_stopping', help='stop when validation loss stops improving', action='store_true')
    parser.add_argument('--train', help='train for given number of epochs, compute and store class means', type=int, default=0)
    parser.add_argument('--validate', help='report loss on validation data', action='store_true')
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    if args.data_path == '':
        print('Using default data path data/')
        data_path = 'data/'
    else:
        print('Overriding data path to {0}'.format(args.data_path))
        data_path = args.data_path

    if args.save_path == '':
        print('Using default save path model/')
        save_directory = 'model/'
    else:
        print('Overriding save path to {0}'.format(args.save_path))
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        save_directory = args.save_path
    if args.early_stopping:
        save_path = save_directory + 'weights_best.h5'
    else:
        save_path = save_directory + 'weights_final.h5'
    means_path = save_directory + 'means.p'

    params = Params()
    train_generator = DataGenerator(data_path + 'train/', params)
    dev_generator = DataGenerator(data_path + 'dev/', params)
    test_generator = DataGenerator(data_path + 'test/', params)
    num_classes = train_generator.num_classes()

    if not args.nz == 0:
        params.latent_size = args.nz
        print('Overriding latent dimension to size {0}'.format(args.nz))

    network = VAE(params)
    vae = network.vae

    def train():
        base_callbacks = Histories()
        tb_callback = keras.callbacks.TensorBoard(log_dir='logs/')
        checkpoint_callback = keras.callbacks.ModelCheckpoint(save_directory + 'weights_best.h5', save_best_only=True, verbose=1)
        callbacks = [base_callbacks, tb_callback, checkpoint_callback]
        if args.early_stopping:
            earlystopping_callback = keras.callbacks.EarlyStopping(verbose=1, patience=10)
            callbacks.append(earlystopping_callback)
        vae.fit_generator(generator=train_generator, validation_data=dev_generator, epochs=args.train, callbacks=callbacks)
        vae.save_weights(save_directory + 'weights_final.h5')
        print('Saved final weights')

    def compute_means():
        print('Encoding input')
        z = network.encoder.predict_generator(train_generator, verbose=1)

        print('Grouping vectors by class label')
        z_grouped = {}
        for i in range(num_classes):
            z_grouped[i] = []
        n = len(train_generator.generator)
        for i in range(n):
            print('{0} of {1}'.format(i+1, n), end='\r')
            _, y = train_generator.generator[i]
            for j, class_index in enumerate(y):
                z_grouped[class_index].append(z[params.batch_size * i + j])

        class_names = train_generator.class_names()
        class_means = np.zeros((num_classes, params.latent_size))
        print('Computing class means')
        for i in range(num_classes):
            print('{0} of {1}'.format(i+1, num_classes), end='\r')
            if len(z_grouped[i]) > 0:
                class_means[i] = np.mean(z_grouped[i], axis=0)

        with open(means_path, 'wb') as f:
            pickle.dump(class_means, f)
        print('Successfully wrote means to file: {0}'.format(means_path))

    if args.train:
        train()
        compute_means()

    if args.validate:
        vae.load_weights(save_directory + 'weights_best.h5')
        print('Evaluating best weights')
        loss = vae.evaluate_generator(generator=dev_generator, verbose=1)
        print('Best weights validation loss: {0}'.format(loss))
        if os.path.exists(save_directory + 'weights_final.h5'):
            vae.load_weights(save_directory + 'weights_final.h5')
            print('Evaluating final weights')
            loss = vae.evaluate_generator(generator=dev_generator, verbose=1)
            print('Final weights validation loss: {0}'.format(loss))
