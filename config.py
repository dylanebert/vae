import os
from vae import VAE

class Config:
    image_size = 32
    filters = 64
    latent_size = 300
    batch_size = 100
    learning_rate = .001
    data_path = 'data/'
    model_path = 'model/'
    train_path = os.path.join(data_path, 'train')
    dev_path = os.path.join(data_path, 'dev')
    test_path = os.path.join(data_path, 'test')
    weights_path = os.path.join(model_path, 'weights_best.h5')
    overfit_path = os.path.join(model_path, 'weights_overfit.h5')
    log_path = os.path.join(model_path, 'logs')
    encodings_path = os.path.join(model_path, 'encodings.p')
    means_path = os.path.join(model_path, 'means.p')
    self_path = os.path.join(model_path, 'config.p')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    trained = os.path.exists(weights_path)
    computed_encodings = os.path.exists(encodings_path)
    computed_means = os.path.exists(means_path)

config = Config()
vae = VAE(config)
vae.compute_encodings()
