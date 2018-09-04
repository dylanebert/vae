# Variational Autoencoder for UTZappos50K Shoe Dataset

## To run:
- Download and install prerequisites [Tensorflow](https://www.tensorflow.org/install/) and [Keras](https://keras.io/)
- Download your desired image dataset
- Restructure the dataset into the directory structure used by Keras' [flow_from_directory](https://keras.io/preprocessing/image/) util. Use utils/splitutil.py to help with this.
- Run vae.py with desired arguments. For example, to train for 10 epochs, use 'python vae.py --data_path /mydatapath/ --train 10'. For a full list of arguments, use 'python vae.py -h'.

## Architecture:
- **vae.py** contains everything for the actual autoencoder, using command line arguments for different tasks
  - By default, train results are saved in model/, which is restored for all inference tasks. This can be changed with the --save_path argument.
- **data_generator.py** is a wrapper for Keras data generators
- **params.py** contains the hyperparameters used by vae.py as follows:
  - image_size: square size of the input images (e.g. 28 for 28x28 images)
  - num_channels: number of input image channels (1 for grayscale, 3 for rgb)
  - latent_size: dimension of the latent space (very important, more info at this [tutorial](http://kvfrans.com/variational-autoencoders-explained/))
  - learning_rate: learning rate. increase to train faster, at risk of poor results.
  - batch_size: batch size. mostly just affects training speed.
  - filters: number of convolution filters to apply.
