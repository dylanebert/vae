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
from scipy.spatial.distance import cosine, euclidean
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class VAE:
    def __init__(self, config):
        self.config = config
        self.train_generator = None
        self.dev_generator = None
        self.test_generator = None
        self.data_loaded = False
        self.encodings = {}
        self.test_encodings = {}
        self.class_means = {}

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

        if config.trained:
            self.vae.load_weights(config.weights_path)
            print('Loaded weights')

    def build_generators(self):
        self.train_generator = DataGenerator(self.config.train_path, self.config.image_size, self.config.batch_size)
        self.dev_generator = DataGenerator(self.config.dev_path, self.config.image_size, self.config.batch_size)
        self.test_generator = DataGenerator(self.config.test_path, self.config.image_size, self.config.batch_size)
        self.data_loaded = True

    def train(self, max_epochs=1000, overfit=False):
        if not self.data_loaded:
            self.build_generators()
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.config.log_path)
        checkpoint_callback = keras.callbacks.ModelCheckpoint(self.config.weights_path, save_best_only=True, verbose=1)
        callbacks = [tensorboard_callback, checkpoint_callback]
        if not overfit:
            earlystopping_callback = keras.callbacks.EarlyStopping(verbose=1, patience=2)
            callbacks.append(earlystopping_callback)
        self.vae.fit_generator(generator=self.train_generator, validation_data=self.dev_generator, epochs=max_epochs, callbacks=callbacks)
        self.vae.save_weights(self.config.overfit_path)
        self.config.trained = True

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
                    self.encodings[class_name] = []
                self.encodings[class_name].append(z[self.config.batch_size * i + j].tolist())
        with open(self.config.encodings_path, 'wb+') as f:
            pickle.dump(self.encodings, f)
        self.config.computed_encodings = True

    def compute_test_encodings(self):
        if not self.data_loaded:
            self.build_generators()
        z = self.encoder.predict_generator(self.test_generator, verbose=1)
        class_index_dict = self.test_generator.generator.class_indices
        index_class_dict = {k: v for v, k in class_index_dict.items()}
        filenames = self.test_generator.generator.filenames
        num_classes = len(index_class_dict)
        n = len(self.test_generator)
        print('Computing encodings')
        for i in range(n):
            print('{0} of {1}'.format(i+1, n), end='\r')
            _, y = self.test_generator.generator[i]
            for j, class_index in enumerate(y):
                class_name = index_class_dict[class_index]
                if class_name not in self.test_encodings:
                    self.test_encodings[class_name] = {'encodings': [], 'filenames': []}
                self.test_encodings[class_name]['encodings'].append(z[self.config.batch_size * i + j].tolist())
                self.test_encodings[class_name]['filenames'].append(filenames[self.config.batch_size * i + j])
        with open(self.config.test_encodings_path, 'wb+') as f:
            pickle.dump(self.test_encodings, f)
        self.config.computed_test_encodings = True

    def compute_means(self):
        if not self.config.computed_encodings:
            self.compute_encodings()
        else:
            with open(config.encodings_path, 'rb') as f:
                self.encodings = pickle.load(f)
            print('Loaded encodings')
        print('Computing class means')
        self.class_means = {}
        for label, encodings in self.encodings.items():
            self.class_means[label] = np.mean(encodings, axis=0).tolist()
        with open(self.config.means_path, 'wb+') as f:
            pickle.dump(self.class_means, f)
        self.config.computed_means = True

    def decode_means(self):
        if not self.config.computed_means:
            self.compute_means()
        print('Decoding class means')
        class_names = list(self.class_means.keys())
        class_means = np.array(list(self.class_means.values()))
        decoded = self.generator.predict(class_means)
        for i, img in enumerate(decoded):
            class_name = class_names[i]
            image_path = os.path.join(self.config.image_path, class_name + '.jpg')
            imsave(image_path, img)

    def predict(self):
        if not self.config.computed_means:
            self.compute_means()
        else:
            with open(config.means_path, 'rb') as f:
                self.class_means = pickle.load(f)
            print('Loaded means')
        if not self.config.computed_encodings:
            self.compute_encodings()
        else:
            with open(config.test_encodings_path, 'rb') as f:
                self.test_encodings = pickle.load(f)
            print('Loaded test encodings')
        print('Predicting')
        means = self.class_means
        encodings = self.test_encodings
        k = 0
        n = len(encodings)
        for label, entry in encodings.items():
            encs = entry['encodings']
            filenames = entry['filenames']
            path = os.path.join(self.config.predictions_path, label + '.json')
            if os.path.exists(path):
                os.remove(path)
            k += 1
            print('{0} of {1}'.format(k, n), end='\r')
            for i, enc in enumerate(encs):
                try:
                    nearest_cos = list(dict(sorted(means.items(), key=lambda x: cosine(enc, x[1]))[:100]).keys())
                    nearest_euc = list(dict(sorted(means.items(), key=lambda x: euclidean(enc, x[1]))[:100]).keys())
                    line = json.dumps({'label': label, 'filename': filenames[i], 'cos': cosine(enc, means[label]), 'euc': euclidean(enc, means[label]), 'predictions_cos': nearest_cos, 'nearest_euc': nearest_euc})
                    with open(path, 'a+') as f:
                        f.write('{0}\n'.format(line))
                except:
                    print('Failed on {0}'.format(label))
        self.config.predicted = True
