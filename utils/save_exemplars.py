import json
import os
from shutil import copyfile

from_path = '/data/gmc/train'

to_path = '/data/gmc_exemplars/nearest'
with open('model/gmc/exemplars_nearest', 'r') as f:
    for line in f:
        data = json.loads(line)
        copyfile(os.path.join(from_path, data['filename']), os.path.join(to_path, data['label'] + '.jpg'))

to_path = '/data/gmc_exemplars/random'
with open('model/gmc/exemplars_random', 'r') as f:
    for line in f:
        data = json.loads(line)
        copyfile(os.path.join(from_path, data['filename']), os.path.join(to_path, data['label'] + '.jpg'))
