#!/bin/bash

if [ $# -ne 2 ]
then 
  echo "Usage: prepare_phenobench.sh <phenobench_dir> <export_dir>"
  echo "Please try again."
  exit 1
fi 

mkdir -p $2

cp -r $1/{train,val}/* $2
cp ./split.yaml $2