from vae import VAE
from config import Config
import argparse
import os
import sys

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
parser.add_argument('--compute_test_encodings', help='compute and store train encodings', action='store_true')
parser.add_argument('--compute_means', help='compute and store mean encoding of each word', action='store_true')
parser.add_argument('--decode_means', help='decode and store image corresponding to each mean', action='store_true')
parser.add_argument('--predict', help='predict each label according to nearest means (euc and cos) and save', action='store_true')
parser.add_argument('--compute_reduced', help='compute reduced encodings and means, for visualization', action='store_true')
parser.add_argument('-a', '--all', help='shorthand to perform all training procedures', action='store_true')
args = parser.parse_args()
if not args.train and not args.compute_encodings and not args.compute_test_encodings and not args.compute_means and not args.decode_means and not args.predict and not args.compute_reduced and not args.all:
    sys.exit('Error: at least one command is required (train/compute_encodings/compute_means/decode_means/compute_reduced/predict/all)')

if args.data_path is not None and not os.path.exists(args.data_path):
    sys.exit('Error: data path {0} not found'.format(args.data_path))
config = Config(args.data_path, args.model_path, args.image_size, args.filters, args.latent_size, args.batch_size, args.learning_rate)
if args.model_path is not None and os.path.exists(os.path.join(args.model_path, 'config.json')):
    config.load(os.path.join(args.model_path, 'config.json'))

if args.compute_encodings:
    config.computed_encodings = False
if args.compute_test_encodings:
    config.computed_test_encodings = False
if args.compute_means:
    config.computed_means = False
if args.predict:
    config.predicted = False
if args.compute_reduced:
    config.computed_reduced = False

vae = VAE(config)
if args.train is not 0:
    vae.train(args.train)
elif args.all:
    vae.train()
if args.compute_encodings or args.all:
    vae.compute_encodings()
if args.compute_test_encodings or args.all:
    vae.compute_test_encodings()
if args.compute_means or args.all:
    vae.compute_means()
if args.decode_means or args.all:
    vae.decode_means()
if args.predict or args.all:
    vae.predict()
if args.compute_reduced or args.all:
    vae.compute_reduced()
config.save()
