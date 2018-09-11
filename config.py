import os

class Config:
    def __init__(self, image_size=32, filters=64, latent_size=300, batch_size=100, learning_rate=.001, data_path='data/', model_path='model/'):
        self.image_size = image_size
        self.filters = filters
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_path = data_path
        self.model_path = model_path
        self.train_path = os.path.join(data_path, 'train')
        self.dev_path = os.path.join(data_path, 'dev')
        self.test_path = os.path.join(data_path, 'test')
        self.weights_path = os.path.join(model_path, 'weights_best.h5')
        self.overfit_path = os.path.join(model_path, 'weights_overfit.h5')
        self.log_path = os.path.join(model_path, 'logs')
        self.encodings_path = os.path.join(model_path, 'encodings.p')
        self.means_path = os.path.join(model_path, 'means.p')
        self.save_path = os.path.join(model_path, 'config.p')
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.trained = False
        self.computed_encodings = False
        self.computed_means = False
