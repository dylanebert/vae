# Variational Autoencoder for UTZappos50K Shoe Dataset

## To run:
- Download and install prerequisites [Tensorflow](https://www.tensorflow.org/install/) and [Keras](https://keras.io/)
- Download the [UTZap50K](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/) image dataset. Download and extract 'ut-zap50k-images-square' to the data folder you created.
- Restructure dataset into the directory structure used by Keras' [flow_from_directory](https://keras.io/preprocessing/image/) util. You can use splitutil.py to help with this.
- Run vae.py with desired arguments. For example, to train for 10 epochs and display image reconstructions, use 'python vae.py --train 10 --reconstruct'. For a full list of arguments, use 'python vae.py -h'.

## Architecture:
- **vae.py** contains everything for the actual autoencoder, using command line arguments for different tasks
<<<<<<< HEAD
  - By default, train results are saved in model/model.h5, which is restored for all inference tasks. This can be changed with the --save argument.
- **data_loader.py** is called by vae.py to provide image data. This can be substituted for another data loader, so long as the data is provided in the shape [num_images, image_size, image_size, num_channels], and the corresponding values are changed in the hyperparameters file.
- **hyperparams.json** contains the hyperparameters used by vae.py:
  - be sure to change image size, channels, and classes if using a different dataset
=======
- **zappos_loader.py** is called by vae.py to provide image data. This can be substituted for another data loader, so long as the data is provided in the shape [num_images, image_size, image_size, num_channels], and the corresponding values are changed in the hyperparameters file.
- **hyperparams.json** contains the hyperparameters used by vae.py as follows:
  -image_size: square size of the input images (e.g. 28 for 28x28 images)
  -n_z: dimension of the latent space (very important, more info at this [tutorial](http://kvfrans.com/variational-autoencoders-explained/))
  - learning_rate: learning rate. increase to train faster, at risk of poor results.
  - batch_size: batch size. mostly just affects training speed.
  - kernel_size: size of convolution filters (e.g. 3 will apply filters of size 3x3).
  - filters: number of convolution filters to apply.
  - channels: number of input image channels (1 for grayscale, 3 for rgb)
  - epsilon_std: standard deviation of latent space sampling. affects smoothness of latent space.
  
>>>>>>> 59e8833... Update README.md
