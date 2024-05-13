#!/bin/bash

if ! command -v wget > /dev/null 2>&1
then 
  echo "Please install wget on your machine"
  exit 1
fi

if [ $# -ne 1 ]
then 
  echo "Usage: get_phenobench.sh <download_dir>"
  echo "Please try again."
  exit 1
fi

if [ ! -d $1 ]
then 
  echo "mkdir"
  mkdir $1
fi

cp ./split.yaml $1

cd $1
wget https://www.phenobench.org/data/PhenoBench-v110.zip && unzip -d data PhenoBench-v110.zip && rm PhenoBench-v110.zip

cp -r $1/data/PhenoBench/{train,val}/* $1
cp -r $1/data/PhenoBench/test $1

rm -rf $1/data