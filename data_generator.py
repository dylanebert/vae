import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator(keras.utils.Sequence):
    def __init__(self, path, image_size, batch_size):
        datagen = ImageDataGenerator(rescale=1./255)
        self.generator = datagen.flow_from_directory(path, target_size=(image_size, image_size), color_mode='rgb', batch_size=batch_size, shuffle=True, class_mode='sparse')
        self.train_mode = True

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        X, y = self.generator[index]

        if self.train_mode:
            return X, []
        else:
            return X, y
