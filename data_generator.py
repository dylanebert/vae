import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator(keras.utils.Sequence):
    def __init__(self, path, image_size, batch_size, train=False):
        datagen = ImageDataGenerator(rescale=1./255)
        self.generator = datagen.flow_from_directory(path, target_size=(image_size, image_size), color_mode='rgb', batch_size=batch_size, shuffle=train, class_mode='sparse')

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        X, y = self.generator[index]
        return X, []

if __name__ == '__main__':
    #To test runtime
    import datetime
    print(datetime.datetime.now())
    train_generator = DataGenerator('/data/nlp/gmc/train', 64, 100)
    dev_generator = DataGenerator('/data/nlp/gmc/dev', 64, 100)
    test_generator = DataGenerator('/data/nlp/gmc/test', 64, 100)
    print(datetime.datetime.now())
