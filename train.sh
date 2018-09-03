#!/bin/bash
source /home/debert/tensorflow-gpu/bin/activate
export CUDA_HOME=/contrib/projects/cuda8.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export PATH=${CUDA_HOME}/bin:${PATH}
cd /home/debert/vae
python3 vae.py --data_path /data/people/debert/mmid_concrete_nouns_split/ --train 100
