#!/bin/bash
python /home/dylan/Documents/vae/vae.py --data_path /data/flickr30k-images-data/ --save_path model/nz10/ --early_stopping --nz 10 --train 100
python /home/dylan/Documents/vae/vae.py --data_path /data/flickr30k-images-data/ --save_path model/nz25/ --early_stopping --nz 25 --train 100
python /home/dylan/Documents/vae/vae.py --data_path /data/flickr30k-images-data/ --save_path model/nz50/ --early_stopping --nz 50 --train 100
python /home/dylan/Documents/vae/vae.py --data_path /data/flickr30k-images-data/ --save_path model/nz75/ --early_stopping --nz 75 --train 100
python /home/dylan/Documents/vae/vae.py --data_path /data/flickr30k-images-data/ --save_path model/nz100/ --early_stopping --nz 100 --train 100
