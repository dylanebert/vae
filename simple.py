import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import optimizers
from keras import metrics
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.models import Model
from keras.datasets import cifar10
from params import Params
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='train for given number of epochs', type=int, default=0)
args = parser.parse_args()

params = Params()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.reshape(x_train, (-1, params.image_size * params.image_size * params.num_channels)).astype('float32') / 255.
x_test = np.reshape(x_test, (-1, params.image_size * params.image_size * params.num_channels)).astype('float32') / 255.

x = Input(shape=(params.image_size * params.image_size * params.num_channels,))
z_mean = Dense(params.latent_size, name='z_mean')(x)
z_stddev = Dense(params.latent_size, name='z_log_var')(x)

#random sampling from Z
def sampling(args):
    z_mean, z_stddev = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], params.latent_size), mean=0., stddev=1.0)
    return z_mean + K.exp(z_stddev) * epsilon

z = Lambda(sampling, output_shape=(params.latent_size,), name='z')([z_mean, z_stddev])

encoder = Model(x, z)

decoder_output = Dense(params.image_size * params.image_size * params.num_channels, activation='sigmoid')

x_reconstruct = decoder_output(z)
vae = Model(x, x_reconstruct)

vae = Model(x, x_reconstruct)
vae.summary()

xent_loss = params.image_size * params.image_size * metrics.binary_crossentropy(x, x_reconstruct)
kl_loss = -0.5 * K.sum(1 + z_stddev - K.square(z_mean) - K.exp(z_stddev), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
vae.add_loss(vae_loss)

optimizer = optimizers.Adam(lr=params.learning_rate)
vae.compile(optimizer=optimizer)
vae.summary()

_z = Input(shape=(params.latent_size,))
_x_reconstruct = decoder_output(_z)
decoder = Model(_z, _x_reconstruct)

if args.train:
    tb_callback = keras.callbacks.TensorBoard(log_dir='logs/')
    checkpoint_callback = keras.callbacks.ModelCheckpoint('test/weights_best.h5', save_best_only=True, verbose=1)
    #earlystopping_callback = keras.callbacks.EarlyStopping(verbose=1, patience=10)
    vae.fit(x_train, epochs=args.train, batch_size=params.batch_size, validation_split=0.2, callbacks=[tb_callback, checkpoint_callback])
    vae.save_weights('test/weights_final.h5')

try:
    vae.load_weights('test/weights_best.h5')
    print('Model restored')
except:
    print('Failed to restore weights')

batch = x_train[:params.batch_size]
reconstructed = vae.predict(batch)

plt.figure(figsize=(6, 10))
for i in range(5):
    x_input = np.reshape(batch[i], (params.image_size, params.image_size, params.num_channels))
    subplot = plt.subplot(5, 2, 2 * i + 1)
    subplot.imshow(x_input)
    if i == 0:
        plt.title('Original image')

    x_reconstruct = np.reshape(reconstructed[i], (params.image_size, params.image_size, params.num_channels))
    subplot = plt.subplot(5, 2, 2 * i + 2)
    subplot.imshow(x_reconstruct)
    if i == 0:
        plt.title('Reconstruction')
plt.suptitle('Training data')
plt.show()
