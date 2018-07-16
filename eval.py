import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import os
import argparse
from vae import VAE
from params import Params
import pickle
from matplotlib import pyplot as plt
from data_generator import DataGenerator
from tabulate import tabulate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='override directory for data', type=str, default='')
parser.add_argument('--save_path', help='override path to save files', type=str, default='')
parser.add_argument('--nz', help='override latent dimension hyperparameter', type=int, default=0)
parser.add_argument('--early_stopping', help='use early stopping weights over final weights', action='store_true')
parser.add_argument('--reconstruct', help='reconstruct training images', action='store_true')
parser.add_argument('--sim', help='print class similarity metrics', action='store_true')
parser.add_argument('--img2txt', help='evaluate image to text (image tagging)', action='store_true')
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

if args.sim:
    vae.load_weights(save_path)
    print('Model restored')

    class_means = pickle.load(open(means_path, 'rb'))
    class_names = train_generator.class_names()

    print('Decoding class means')
    decoded = network.generator.predict(class_means, batch_size=params.batch_size, verbose=1)

    table = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            z_dist = np.linalg.norm(class_means[i] - class_means[j])
            img_dist = np.mean(np.absolute(decoded[i] - decoded[j]) * 255.)
            cos = cosine_similarity([class_means[i], class_means[j]])[0, 1]
            kl = np.sum(entropy(decoded[i], decoded[j], 2))
            table.append([class_names[i], class_names[j], z_dist, img_dist, cos, kl])
    print(tabulate(table, headers=['class a', 'class b', 'z_dist', 'img_dist', 'cossim', 'kl-div']))

if args.img2txt:
    vae.load_weights(save_path)
    print('Model restored')

    class_means = pickle.load(open(means_path, 'rb'))
    class_names = train_generator.class_names()

    z = network.encoder.predict_generator(test_generator, verbose=1)
    correct = np.zeros((z.shape[0]))
    for i in range(len(test_generator.generator)):
        _, y = test_generator.generator[i]
        for j, label in enumerate(y):
            correct[i * params.batch_size + j] = label

    predicted = np.zeros((z.shape[0]))
    for i, z_i in enumerate(z):
        dists = [np.linalg.norm(class_mean - z_i) for class_mean in class_means]
        min_dist = np.argmin(dists)
        predicted[i] = min_dist

    cm = confusion_matrix(correct, predicted)
    print('Confusion matrix:')
    print(cm)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names.values(), rotation=45)
    plt.yticks(tick_marks, class_names.values())
    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.show()
