#!/bin/bash 

if ! command -v conda > /dev/null 2>&1
then 
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
fi 

conda create -n pspa python=3.9

eval "$(conda shell.bash hook)"
conda activate pspa

pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu11==24.4.* cuml-cu11==24.4.*

pip install -r requirements.txt