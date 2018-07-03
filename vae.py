import tensorflow as tf
import keras
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.models import model_from_json
from keras import optimizers
from keras import backend as K
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
import callbacks
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
parser.add_argument('--save_path', help='override path to save files', type=str, default='')
parser.add_argument('--nz', help='override default latent dimension from hyperparameter file', type=int, default=0)
parser.add_argument('--early_stopping', help='stop when validation loss stops improving; use best weights over final weights', action='store_true')
parser.add_argument('--grayscale', help='convert data to grayscale', action='store_true')
parser.add_argument('--train', help='train for given number of epochs', type=int, default=0)
parser.add_argument('--validate', help='report loss on validation data', action='store_true')
parser.add_argument('--reconstruct', help='show reconstructions of input', action='store_true')
parser.add_argument('--random', help='show generations from random sampling of the latent space', action='store_true')
parser.add_argument('--manifold', help='show 2d manifold', action='store_true')
parser.add_argument('--plot', help='plot test class labels', action='store_true')
parser.add_argument('--compute_means', help='compute mean (prototype) of each class', action='store_true')
parser.add_argument('--test_means', help='test classification accuracy using nearest means', action='store_true')
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

#load hyperparameters from file
params = json.load(open(hyperparams_path))
print('Successfully loaded hyperparameters: {0}'.format(params))

#create data loaders
train_generator = DataGenerator(data_path + 'train/', params, grayscale=args.grayscale)
dev_generator = DataGenerator(data_path + 'dev/', params, grayscale=args.grayscale)
test_generator = DataGenerator(data_path + 'test/', params, grayscale=args.grayscale)

num_classes = train_generator.num_classes()
if args.grayscale:
    num_channels = 1
    cmap = 'gray'
else:
    num_channels = 3
    cmap = None
#hidden_size = params['image_size'] * params['image_size'] * num_channels

if not args.nz == 0:
    params['n_z'] = args.nz
    print('Overriding latent dimension to size {0}'.format(args.nz))

#input
input_shape = [params['image_size'], params['image_size'], num_channels]
x = Input(shape=input_shape)

#encoder
conv1 = Conv2D(num_channels, kernel_size=(2, 2), padding='same', activation='relu')(x)
conv2 = Conv2D(params['filters'], kernel_size=(2, 2), padding='same', activation='relu', strides=(2, 2))(conv1)
conv3 = Conv2D(params['filters'], kernel_size=params['kernel_size'], padding='same', activation='relu', strides=1)(conv2)
conv4 = Conv2D(params['filters'], kernel_size=params['kernel_size'], padding='same', activation='relu', strides=1)(conv3)
flat = Flatten()(conv4)
#hidden = Dense(hidden_size, activation='relu')(flat)

#latent space Z (mean and std)
z_mean = Dense(params['n_z'])(flat)
z_stddev = Dense(params['n_z'])(flat)

#random sampling from Z
def sampling(args):
    z_mean, z_stddev = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], params['n_z']), mean=0., stddev=params['epsilon_std'])
    return z_mean + K.exp(z_stddev) * epsilon

z = Lambda(sampling, output_shape=(params['n_z'],))([z_mean, z_stddev])

#decoder
#decoder_hidden = Dense(hidden_size, activation='relu')
decoder_upsample = Dense(params['filters'] * (params['image_size'] // 2) * (params['image_size'] // 2), activation='relu')

output_shape = (params['batch_size'], params['image_size'] // 2, params['image_size'] // 2, params['filters'])

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv1 = Conv2DTranspose(params['filters'], kernel_size=params['kernel_size'], padding='same', strides=1, activation='relu')
decoder_deconv2 = Conv2DTranspose(params['filters'], kernel_size=params['kernel_size'], padding='same', strides=1, activation='relu')

output_shape = (params['batch_size'], params['filters'], params['image_size'] + 1, params['image_size'] + 1)

decoder_deconv3_upsamp = Conv2DTranspose(params['filters'], kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')
decoder_mean_squash = Conv2D(num_channels, kernel_size=2, padding='valid', activation='sigmoid')

#hidden_decoded = decoder_hidden(z)
up_decoded = decoder_upsample(z)
reshape_decoded = decoder_reshape(up_decoded)
deconv1_decoded = decoder_deconv1(reshape_decoded)
deconv2_decoded = decoder_deconv2(deconv1_decoded)
x_decoded_relu = decoder_deconv3_upsamp(deconv2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

#Model for full vae
vae = Model(x, x_decoded_mean_squash)

#Loss function
xent_loss = params['image_size'] * params['image_size'] * metrics.binary_crossentropy(K.flatten(x), K.flatten(x_decoded_mean_squash))
kl_loss = -0.5 * K.sum(1 + z_stddev - K.square(z_mean) - K.exp(z_stddev), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
vae.add_loss(vae_loss)

optimizer = optimizers.Adam(lr=params['learning_rate'])
vae.compile(optimizer=optimizer)
vae.summary()

#model to project input onto latent space
encoder = Model(x, z_mean)

#model for generating image from latent vector
decoder_input = Input(shape=(params['n_z'],))
#_hidden_decoded = decoder_hidden(decoder_input)
_up_decoded = decoder_upsample(decoder_input)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv1_decoded = decoder_deconv1(_reshape_decoded)
_deconv2_decoded = decoder_deconv2(_deconv1_decoded)
_x_decoded_relu = decoder_deconv3_upsamp(_deconv2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)

#train
if args.train:
    base_callbacks = callbacks.Histories()
    tb_callback = keras.callbacks.TensorBoard(log_dir='logs/')
    checkpoint_callback = keras.callbacks.ModelCheckpoint(save_directory + 'weights_best.h5', save_best_only=True, verbose=1)
    callbacks = [base_callbacks, tb_callback, checkpoint_callback]
    if args.early_stopping:
        earlystopping_callback = keras.callbacks.EarlyStopping(verbose=1, patience=10)
        callbacks.append(earlystopping_callback)
    vae.fit_generator(generator=train_generator, validation_data=dev_generator, epochs=args.train, callbacks=callbacks)
    vae.save_weights(save_directory + 'weights_final.h5')
    print('Saved final weights')

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

#reconstruct
if args.reconstruct:
    vae.load_weights(save_path)
    print('Model restored')

    batch_train, _ = train_generator[0]
    reconstructed_train = vae.predict(batch_train, batch_size=params['batch_size'], verbose=1)

    batch_test, _ = test_generator[0]
    reconstructed_test = vae.predict(batch_test, batch_size=params['batch_size'], verbose=1)

    plt.figure(figsize=(6, 10))
    for i in range(5):
        x_input = batch_train[i]
        subplot = plt.subplot(5, 2, 2 * i + 1)
        if args.grayscale:
            x_input = x_input[:,:,0]
        subplot.imshow(x_input, cmap=cmap)
        if i == 0:
            plt.title('Original image')

        x_reconstruct = reconstructed_train[i]
        subplot = plt.subplot(5, 2, 2 * i + 2)
        if args.grayscale:
            x_reconstruct = x_reconstruct[:,:,0]
        subplot.imshow(x_reconstruct, cmap=cmap)
        if i == 0:
            plt.title('Reconstruction')
    plt.suptitle('Training data')
    plt.show()

    plt.figure(figsize=(6, 10))
    for i in range(5):
        x_input = batch_test[i]
        subplot = plt.subplot(5, 2, 2 * i + 1)
        if args.grayscale:
            x_input = x_input[:,:,0]
        subplot.imshow(x_input, cmap=cmap)
        if i == 0:
            plt.title('Original image')

        x_reconstruct = reconstructed_test[i]
        subplot = plt.subplot(5, 2, 2 * i + 2)
        if args.grayscale:
            x_reconstruct = x_reconstruct[:,:,0]
        subplot.imshow(x_reconstruct, cmap=cmap)
        if i == 0:
            plt.title('Reconstruction')
    plt.suptitle('Test data')
    plt.show()

#random
if args.random:
    vae.load_weights(save_path)
    print('Model restored')

    z_sample = np.random.normal(size=[params['batch_size'], params['n_z']])
    x_decoded = generator.predict(z_sample, batch_size=params['batch_size'], verbose=1)

    plt.figure()
    for i in range(3):
        for j in range(3):
            img = x_decoded[i * 3 + j].reshape(params['image_size'], params['image_size'], num_channels)
            subplot = plt.subplot(3, 3, i * 3 + j + 1)
            if args.grayscale:
                img = img[:,:,0]
            subplot.imshow(img, cmap=cmap)
    plt.suptitle('Random sampling from Z')
    plt.show()

#manifold (for 2d latent space)
if args.manifold:
    vae.load_weights(save_path)
    print('Model restored')

    n = 15
    figure = np.zeros((n * params['image_size'], n * params['image_size'], num_channels))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    z_sample = np.zeros(shape=[n*n, params['n_z']])
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample[i * n + j] = [xi, yi]
    x_decoded = generator.predict(z_sample, batch_size=n*n, verbose=1)
    for i in range(n):
        for j in range(n):
            img = x_decoded[i * n + j].reshape(params['image_size'], params['image_size'], num_channels)
            figure[i * params['image_size'] : (i + 1) * params['image_size'], j * params['image_size'] : (j + 1) * params['image_size']] = img
    plt.figure(figsize=(10, 10))
    if args.grayscale:
        figure = figure[:,:,0]
    plt.imshow(figure, cmap=cmap)
    plt.show()

#plot
if args.plot:
    vae.load_weights(save_path)
    print('Model restored')

    labels = test_generator.get_labels()

    print('Encoding input')
    num_batches = 10
    plt.figure()
    for i in range(num_batches):
        X, _ = test_generator[i]
        y = labels[i * params['batch_size'] : i * params['batch_size'] + params['batch_size']]
        z = encoder.predict(X, verbose=0)
        plt.scatter(z[:, 0], z[:, 1], c=y, s=3, cmap=cm.afmhot)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()

def compute_means(filename):
    print('Encoding input')
    encoded = encoder.predict_generator(train_generator, verbose=1)
    labels = train_generator.get_labels()

    print('Grouping vectors by class label')
    labeled_dict = {}
    for i in range(num_classes):
        labeled_dict[i] = []
    for i, z in enumerate(encoded):
        labeled_dict[labels[i]].append(z)

    class_means = np.zeros((num_classes, params['n_z']))
    for i in range(num_classes):
        if len(labeled_dict[i]) > 0:
            class_means[i] = np.mean(labeled_dict[i], axis=0)
    print('Class means: {0}'.format(class_means))

    with open(filename, 'wb') as f:
        pickle.dump(class_means, f)
    print('Successfully wrote means to file: {0}'.format(filename))

if args.compute_means:
    vae.load_weights(save_path)
    print('Model restored')
    compute_means(means_path)

if args.test_means:
    vae.load_weights(save_path)
    print('Model restored')

    filename = means_path
    try:
        f = open(filename, 'rb')
    except:
        compute_means(filename)
        f = open(filename, 'rb')

    class_means = pickle.load(f)

    print('Encoding test data')
    encoded = encoder.predict_generator(test_generator, verbose=1)
    labels = test_generator.get_labels()
    class_names = test_generator.class_names()

    n_correct = np.zeros(num_classes)
    n_total = np.zeros(num_classes)
    for i, z in enumerate(encoded):
        dists = np.sum((class_means - z) ** 2, axis=1)
        nearest = np.argmin(dists)
        if nearest == labels[i]:
            n_correct[labels[i]] += 1
        n_total[labels[i]] += 1
    accuracy = n_correct / n_total
    print('Reporting class-wise accuracy:')
    for i in range(num_classes):
        print('{0}: {1}'.format(class_names[i], accuracy[i]))
    print('Overall accuracy: {0}'.format(np.mean(accuracy)))

    y_true = []
    y_pred = []
    for i, z in enumerate(encoded):
        dists = np.sum((class_means - z) ** 2, axis=1)
        nearest = np.argmin(dists)
        y_true.append(labels[i])
        y_pred.append(nearest)
    print('Printing confusion matrix for classes {0}'.format([class_names[i] for i in range(num_classes)]))
    print(confusion_matrix(y_true, y_pred))

if args.entailment:
    vae.load_weights(save_path)
    print('Model restored')

    filename = means_path
    try:
        f = open(filename, 'rb')
    except:
        compute_means(filename)
        f = open(filename, 'rb')

    class_indices = train_generator.get_indices()
    class_means = pickle.load(f)
    z = np.zeros((params['batch_size'], params['n_z']))
    labels = ['Boots', 'Sandals', 'Shoes', 'Slippers']
    for i, label in enumerate(labels):
        z[i] = class_means[class_indices[label]]
    reconstructed = generator.predict(z, batch_size=params['batch_size'], verbose=1)

    for i in range(len(labels)):
        for j in range(i, len(labels)):
            cross_entropy = K.eval(K.mean(metrics.binary_crossentropy(K.flatten(K.constant(reconstructed[i])), K.flatten(K.constant(reconstructed[j])))))
            print('Cross entropy between {0} and {1}: \n{2}'.format(labels[i], labels[j], cross_entropy))

    plt.figure(figsize=(2, len(labels) * 2))
    for i in range(len(labels)):
        x_reconstruct = reconstructed[i]
        subplot = plt.subplot(len(labels), 1, i + 1)
        if args.grayscale:
            x_reconstruct = x_reconstruct[:,:,0]
        subplot.imshow(x_reconstruct, cmap=cmap)
        plt.title(labels[i])
        plt.axis('off')
    plt.show()
