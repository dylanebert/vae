from shutil import copy2
import os

base_path = '/data/person_dog/'
dir1 = '/data/person_dog_split/train/'
dir2 = '/data/person_dog_split/dev/'
dir3 = '/data/person_dog_split/test/'
for dirname in os.listdir(base_path):
    os.makedirs(dir1 + dirname)
    os.makedirs(dir2 + dirname)
    os.makedirs(dir3 + dirname)
    filenames = os.listdir(base_path + dirname)
    train_len = len(filenames) // 5 * 4
    dev_len = len(filenames) // 10
    train_filenames = filenames[:train_len]
    dev_filenames = filenames[train_len:(train_len+dev_len)]
    test_filenames = filenames[(train_len+dev_len):]
    for filename in train_filenames:
        copy2(base_path + dirname + '/' + filename, dir1 + dirname + '/' + os.path.basename(filename))
    for filename in dev_filenames:
        copy2(base_path + dirname + '/' + filename, dir2 + dirname + '/' + os.path.basename(filename))
    for filename in test_filenames:
        copy2(base_path + dirname + '/' + filename, dir3 + dirname + '/' + os.path.basename(filename))
