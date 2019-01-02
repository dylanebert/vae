from PIL import Image
import os
import numpy as np
import sys
from tqdm import tqdm

data_path = sys.argv[1]
out_path = sys.argv[2]

with open(out_path, 'w+') as f:
    dirs = os.listdir(data_path)
    for i in tqdm(range(len(dirs))):
        dir = dirs[i]
        imgs = os.listdir(os.path.join(data_path, dir))
        for j, imgfile in enumerate(imgs):
            try:
                im = Image.open(os.path.join(data_path, dir, imgfile)).convert('RGB')
                im.verify()
            except:
                f.write('{0}\n'.format(os.path.join(dir, imgfile)))
