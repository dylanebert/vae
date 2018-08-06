from PIL import Image
import os
import imghdr
import numpy as np

data_path = '/home/dylan/data/mmid/combined/'
cleaned_path = '/home/dylan/data/mmid/combined_cleaned'

max_size = 512

dirs = os.listdir(data_path)
for i, dir in enumerate(dirs):
    imgs = os.listdir(os.path.join(data_path, dir))
    for j, imgfile in enumerate(imgs):
        print('Dir {0} of {1}: File {2} of {3}'.format(i, len(dirs), j, len(imgs)))
        if imghdr.what(os.path.join(data_path, dir, imgfile)) == None:
            print('Failed')
            continue
        try:
            im = Image.open(os.path.join(data_path, dir, imgfile)).convert('RGB')
            im.verify()
            extrema = im.convert('L').getextrema()
            if extrema[0] == extrema[1]:
                print('Failed')
                continue
            if im.size[0] > max_size or im.size[1] > max_size:
                im.thumbnail((512, 512))
            dirpath = os.path.join(cleaned_path, dir)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            im.save(os.path.join(dirpath, '{0}.jpg'.format(j)))
            print('Succeeded')
        except:
            print('Failed')
