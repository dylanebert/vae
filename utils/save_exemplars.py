import json
import os
from shutil import copyfile

from_path = '/data/gmc/train'

to_path = '/data/gmc_exemplars/nearest'
with open('exemplars_nearest', 'r') as f:
    for line in f:
        label, filename, dist = line.rstrip().split('\t')
        copyfile(os.path.join(from_path, filename), os.path.join(to_path, label + '.jpg'))

to_path = '/data/gmc_exemplars/random'
with open('exemplars_random', 'r') as f:
    for line in f:
        label, filename, dist = line.rstrip().split('\t')
        copyfile(os.path.join(from_path, filename), os.path.join(to_path, label + '.jpg'))
