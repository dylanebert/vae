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

    def load(self, path):
        self.__dict__ = json.loads(open(path, 'r').read())

    def save(self):
        with open(self.save_path, 'w+') as f:
            f.write(json.dumps(self.__dict__, indent=4))

    def __str__(self):
        return json.dumps(self.__dict__, indent=4)
