import tensorflow as tf
import keras
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.models import model_from_json
from keras import backend as K
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import os
import pickle
from scipy.stats import norm, multivariate_normal
from matplotlib import cm
from data_generator import DataGenerator
from sklearn.metrics import confusion_matrix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#parse args
parser = argparse.ArgumentParser()
parser.add_argument('--hyperparams_path', help='override path to hyperparameters file', type=str, default='')
parser.add_argument('--data_path', help='override directory for data', type=str, default='')
parser.add_argument('--save_path', help='override weights save file, for both saving and loading', type=str, default='')
parser.add_argument('--nz', help='override default latent dimension from hyperparameter file', type=int, default=0)
parser.add_argument('--train', help='train for given number of epochs', type=int, default=0)
parser.add_argument('--entailment', help='test entailment task', action='store_true')
args = parser.parse_args()

if args.hyperparams_path == '':
    print('Using default hyperparams path hyperparams.json')
    hyperparams_path = 'hyperparams.json'
else:
    print('Overriding hyperparams path to {0}'.format(args.hyperparams_path))
    hyperparams_path = args.hyperparams_path

if args.data_path == '':
    print('Using default data path data/')
    data_path = 'data/'
else:
    print('Overriding data path to {0}'.format(args.data_path))
    data_path = args.data_path

if args.save_path == '':
    print('Using default save path model/one_hot.h5')
    save_path = 'model/one_hot.h5'
else:
    print('Overriding save path to {0}'.format(args.save_path))
    save_path = args.save_path

#load hyperparameters from file
params = json.load(open(hyperparams_path))
print('Successfully loaded hyperparameters: {0}'.format(params))

#create data loaders
train_generator = DataGenerator(data_path + 'train/', params)
dev_generator = DataGenerator(data_path + 'dev/', params)
test_generator = DataGenerator(data_path + 'test/', params)

num_classes = train_generator.num_classes()

if not args.nz == 0:
    params['n_z'] = args.nz
    print('Overriding latent dimension to size {0}'.format(args.nz))

#input
x = Input(shape=[params['image_size'], params['image_size'], params['channels']])
c = Input(shape=[num_classes])

#latent space Z (mean and std)
z_mean = Dense(params['n_z'])(c)
z_stddev = Dense(params['n_z'])(c)

#random sampling from Z
def sampling(args):
    z_mean, z_stddev = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], params['n_z']), mean=0., stddev=params['epsilon_std'])
    return z_mean + K.exp(z_stddev) * epsilon

z = Lambda(sampling, output_shape=(params['n_z'],))([z_mean, z_stddev])

#decoder
decoder_hidden = Dense(params['hidden'], activation='relu')
decoder_upsample = Dense(params['filters'] * (params['image_size'] // 2) * (params['image_size'] // 2), activation='relu')

output_shape = (params['batch_size'], params['image_size'] // 2, params['image_size'] // 2, params['filters'])

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv1 = Conv2DTranspose(params['filters'], kernel_size=params['kernel_size'], padding='same', strides=1, activation='relu')
decoder_deconv2 = Conv2DTranspose(params['filters'], kernel_size=params['kernel_size'], padding='same', strides=1, activation='relu')

output_shape = (params['batch_size'], params['filters'], params['image_size'] + 1, params['image_size'] + 1)

decoder_deconv3_upsamp = Conv2DTranspose(params['filters'], kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')
decoder_mean_squash = Conv2D(params['channels'], kernel_size=2, padding='valid', activation='sigmoid')

hidden_decoded = decoder_hidden(z)
up_decoded = decoder_upsample(hidden_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv1_decoded = decoder_deconv1(reshape_decoded)
deconv2_decoded = decoder_deconv2(deconv1_decoded)
x_decoded_relu = decoder_deconv3_upsamp(deconv2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

#Model for full vae
vae = Model(inputs=(x, c), outputs=(x_decoded_mean_squash))

#Loss function
xent_loss = params['image_size'] * params['image_size'] * metrics.binary_crossentropy(K.flatten(x), K.flatten(x_decoded_mean_squash))
kl_loss = -0.5 * K.sum(1 + z_stddev - K.square(z_mean) - K.exp(z_stddev), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
vae.add_loss(vae_loss)

vae.compile(optimizer='rmsprop')
vae.summary()

#model to project input onto latent space
encoder = Model(inputs=(x, c), outputs=(z_mean))

#model for generating image from latent vector
decoder_input = Input(shape=(params['n_z'],))
_hidden_decoded = decoder_hidden(decoder_input)
_up_decoded = decoder_upsample(_hidden_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv1_decoded = decoder_deconv1(_reshape_decoded)
_deconv2_decoded = decoder_deconv2(_deconv1_decoded)
_x_decoded_relu = decoder_deconv3_upsamp(_deconv2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)

#train
if args.train:
    vae.fit_generator(generator=train_generator, validation_data=dev_generator, epochs=args.train)
    vae.save_weights(save_path)
    print('Model saved')

#reconstruct
if args.entailment:
    vae.load_weights(save_path)
    print('Model restored')

    x = np.zeros((max(num_classes, params['batch_size']), params['image_size'], params['image_size'], params['channels']))
    c = np.zeros((max(num_classes, params['batch_size']), num_classes))
    for i in range(num_classes):
        c[i][i] = 1

    x_decoded = vae.predict([x, c], batch_size=params['batch_size'], verbose=1)
    class_indices = train_generator.get_indices()

    labels = ['person', 'grass', 'water', 'dog']
    for i in range(len(labels)):
        for j in range(i, len(labels)):
            cross_entropy = K.eval(K.mean(metrics.binary_crossentropy(K.flatten(K.constant(x_decoded[i])), K.flatten(K.constant(x_decoded[j])))))
            print('Cross entropy between {0} and {1}: \n{2}'.format(labels[i], labels[j], cross_entropy))

    plt.figure(figsize=(2, len(labels) * 2))
    for i, label in enumerate(labels):
        idx = class_indices[label]
        img = x_decoded[idx]
        subplot = plt.subplot(len(labels), 1, i + 1)
        subplot.imshow(img)
        plt.axis('off')
        plt.title(label)
    plt.suptitle('Class reconstructions')
    plt.show()
