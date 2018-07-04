import numpy as np
import keras
import os
import sys
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator(keras.utils.Sequence):
    def __init__(self, directory, params, one_hot=False, grayscale=False):
        if not os.path.exists(directory):
            sys.exit('Please create and populate data directory {0}'.format(directory))
        datagen = ImageDataGenerator(rescale=1./255)
        self.batch_size = params.batch_size
        self.directory = directory
        self.one_hot = one_hot
        if one_hot:
            self.class_mode = 'categorical'
        else:
            self.class_mode = 'sparse'
        if grayscale:
            self.color_mode = 'grayscale'
        else:
            self.color_mode = 'rgb'
        self.generator = datagen.flow_from_directory(directory, target_size=(params.image_size, params.image_size), color_mode=self.color_mode, batch_size=self.batch_size, shuffle=True, class_mode=self.class_mode)
        self.labels = []
        self.labels_cached = False
        self.labels_cached_list = [False] * len(self)

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        X, y = self.generator[index]

        if not self.labels_cached:
            for label in y:
                self.labels.append(label)
            self.labels_cached_list[index] = True
            if False not in self.labels_cached_list:
                self.labels_cached = True

        if self.one_hot:
            return [X, y], []
        else:
            return X, []

    def num_classes(self):
        return len(self.generator.class_indices)

    def get_labels(self):
        if not self.labels_cached:
            for i in range(len(self)):
                print('Retrieving labels {0} of {1}'.format(i+1, len(self)), end='\r')
                _, _ = self[i]
            print('')
        else:
            print('Found cached labels')

        return self.labels

    def class_names(self):
        return {v: k for k, v in self.generator.class_indices.items()}

    def get_indices(self):
        return self.generator.class_indices
