import numpy as np
import tensorflow as tf
import keras
import os
import argparse
from vae import VAE
from params import Params
from matplotlib import pyplot as plt
from data_generator import DataGenerator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='override directory for data', type=str, default='')
parser.add_argument('--save_path', help='override path to save files', type=str, default='')
parser.add_argument('--nz', help='override latent dimension hyperparameter', type=int, default=0)
parser.add_argument('--early_stopping', help='use early stopping weights over final weights', action='store_true')
parser.add_argument('--reconstruct', help='reconstruct training images', action='store_true')
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

if args.reconstruct:
    vae.load_weights(save_path)
    print('Model restored')

    batch, _ = train_generator[0]
    reconstructed = vae.predict(batch, batch_size=params.batch_size, verbose=1)

    plt.figure(figsize=(6, 10))
    for i in range(5):
        x_input = batch[i]
        subplot = plt.subplot(5, 2, 2 * i + 1)
        subplot.imshow(x_input)
        if i == 0:
            plt.title('Original image')

        x_reconstruct = reconstructed[i]
        subplot = plt.subplot(5, 2, 2 * i + 2)
        subplot.imshow(x_reconstruct)
        if i == 0:
            plt.title('Reconstruction')
    plt.suptitle('Training data reconstructions')
    plt.show()

'''if args.reconstruct:
    vae.load_weights(save_path)
    print('Model restored')

    batch_train, _ = train_generator[0]
    reconstructed_train = vae.predict(batch_train, batch_size=params.batch_size, verbose=1)

    batch_test, _ = test_generator[0]
    reconstructed_test = vae.predict(batch_test, batch_size=params.batch_size, verbose=1)

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

    z_sample = np.random.normal(size=[params.batch_size, params.latent_size])
    x_decoded = generator.predict(z_sample, batch_size=params.batch_size, verbose=1)

    plt.figure()
    for i in range(3):
        for j in range(3):
            img = x_decoded[i * 3 + j].reshape(params.image_size, params.image_size, num_channels)
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
    figure = np.zeros((n * params.image_size, n * params.image_size, num_channels))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    z_sample = np.zeros(shape=[n*n, params.latent_size])
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample[i * n + j] = [xi, yi]
    x_decoded = generator.predict(z_sample, batch_size=n*n, verbose=1)
    for i in range(n):
        for j in range(n):
            img = x_decoded[i * n + j].reshape(params.image_size, params.image_size, params.num_channels)
            figure[i * params.image_size : (i + 1) * params.image_size, j * params.image_size : (j + 1) * params.image_size] = img
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
        y = labels[i * params.batch_size : i * params.batch_size + params.batch_size]
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

    class_means = np.zeros((num_classes, params.latent_size))
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
    z = np.zeros((params.batch_size, params.latent_size))
    labels = ['Boots', 'Sandals', 'Shoes', 'Slippers']
    for i, label in enumerate(labels):
        z[i] = class_means[class_indices[label]]
    reconstructed = generator.predict(z, batch_size=params.batch_size, verbose=1)

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
    plt.show()'''
