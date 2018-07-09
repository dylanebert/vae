#!/bin/bash
source /home/debert/tensorflow-gpu/bin/activate
export CUDA_HOME=/contrib/projects/cuda8.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export PATH=${CUDA_HOME}/bin:${PATH}
python3 /home/debert/vae/vae.py --data_path /home/debert/vae/data/ --save_path /home/debert/vae/model/ --train 100
