#!/bin/bash 

if ! command -v conda > /dev/null 2>&1
then 
  echo "Please install conda on your machine"
  exit 1
fi 

conda create -n pspa python=3.9

eval "$(conda shell.bash hook)"
conda activate pspa

pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu11==24.4.* cuml-cu11==24.4.*

pip install -r requirements.txt