import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose
from keras import backend as K
from keras.optimizers import Adam
from keras import metrics
from params import Params
import argparse
import os
import sys
from data_generator import DataGenerator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class VAEGAN:
    def __init__(self, params):
        image_size = params.image_size
        num_channels = params.num_channels
        filters = params.filters
        latent_size = params.latent_size
        batch_size = params.batch_size
        learning_rate = params.learning_rate

        #decoder
        decoder_upsample = Dense(filters * (image_size // 2) * (image_size // 2), activation='relu')
        decoder_reshape = Reshape([image_size // 2, image_size // 2, filters])
        decoder_deconv1 = Conv2DTranspose(filters, kernel_size=3, padding='same', strides=1, activation='relu')
        decoder_deconv2 = Conv2DTranspose(filters, kernel_size=3, padding='same', strides=1, activation='relu')
        decoder_deconv3_upsamp = Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')
        decoder_reconstr = Conv2D(num_channels, kernel_size=2, padding='valid', activation='sigmoid')

        decoder_input = Input(shape=(latent_size,))
        up_decoded = decoder_upsample(decoder_input)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv1_decoded = decoder_deconv1(reshape_decoded)
        deconv2_decoded = decoder_deconv2(deconv1_decoded)
        x_decoded_relu = decoder_deconv3_upsamp(deconv2_decoded)
        x_reconstr = decoder_reconstr(x_decoded_relu)

        self.generator = Model(decoder_input, x_reconstr)

        #discriminator
        discriminator_flatten = Flatten(input_shape=(image_size, image_size, num_channels))
        discriminator_h1 = Dense(512, activation='relu')
        discriminator_h2 = Dense(256, activation='relu')
        discriminator_res = Dense(1, activation='sigmoid')

        discriminator_input = Input(shape=(image_size, image_size, num_channels))
        flatten_discriminated = discriminator_flatten(discriminator_input)
        h1_discriminated = discriminator_h1(flatten_discriminated)
        h2_discriminated = discriminator_h2(h1_discriminated)
        res_discriminated = discriminator_res(h2_discriminated)

        self.discriminator = Model(discriminator_input, res_discriminated)

        #combined
        z = Input(shape=(latent_size,))
        img = self.generator(z)
        self.discriminator.trainable = False
        validity = self.discriminator(img)
        self.gan = Model(z, validity)
        optimizer = Adam(.0002, .5)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)

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

    network = VAEGAN(params)
    gan = network.gan

    def train():
        valid = np.ones((params.batch_size, 1))
        fake = np.zeros((params.batch_size, 1))

        for epoch in range(10):
            for i in range(len(train_generator)):
                #Train discriminator
                imgs = train_generator[i]

                noise = np.random.normal(0, 1, (params.batch_size, params.latent_size))
                gen_imgs = network.generator.predict(noise)

                d_loss_real = network.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = network.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = .5 * np.add(d_loss_real, d_loss_fake)

                #Train generator
                noise = np.random.normal(0, 1, (params.batch_size, params.latent_size))

                g_loss = gan.train_on_batch(noise, valid)
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    train()
