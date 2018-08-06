from shutil import copy2
import os

base_path = '/home/dylan/data/mmid/combined_cleaned/'
dir1 = '/home/dylan/data/mmid/combined_split/train/'
dir2 = '/home/dylan/data/mmid/combined_split/dev/'
dir3 = '/home/dylan/data/mmid/combined_split/test/'
dirs = os.listdir(base_path)
for i, dirname in enumerate(dirs):
    print('{0} of {1}'.format(i + 1, len(dirs)), end='\r')
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
