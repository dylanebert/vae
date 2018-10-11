import os
import json

class Config():
    def __init__(self, data_path=None, model_path=None, image_size=None, filters=None, latent_size=None, batch_size=None, learning_rate=None):
        if data_path is None or model_path is None or image_size is None or filters is None or latent_size is None or batch_size is None or learning_rate is None:
            return
        self.data_path = data_path
        self.model_path = model_path
        self.image_size = image_size
        self.filters = filters
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.train_path = os.path.join(data_path, 'train')
        self.dev_path = os.path.join(data_path, 'dev')
        self.test_path = os.path.join(data_path, 'test')
        self.weights_path = os.path.join(model_path, 'weights_best.h5')
        self.overfit_path = os.path.join(model_path, 'weights_overfit.h5')
        self.log_path = os.path.join(model_path, 'logs')
        self.encodings_path = os.path.join(model_path, 'encodings.p')
        self.test_encodings_path = os.path.join(model_path, 'test_encodings.p')
        self.means_path = os.path.join(model_path, 'means.p')
        self.save_path = os.path.join(model_path, 'config.json')
        self.image_path = os.path.join(model_path, 'images')
        self.encodings_reduced_path = os.path.join(model_path, 'encodings_reduced.p')
        self.means_reduced_path = os.path.join(model_path, 'means_reduced.p')
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)
        self.trained = False
        self.computed_encodings = False
        self.computed_test_encodings = False
        self.computed_means = False
        self.computed_reduced = False

    def load(self, path):
        self.__dict__ = json.loads(open(path, 'r').read())

    def save(self):
        with open(self.save_path, 'w+') as f:
            f.write(json.dumps(self.__dict__, indent=4))

    def __str__(self):
        return json.dumps(self.__dict__, indent=4)
