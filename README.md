# PSPA

*[Panoptic Segmentation with Partial Annotations for Agricultural Robots](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/weyler2024ral.pdf)*

We present a novel approach to leverage partial annotations for panoptic segmentation.
These partial annotations contain
ground truth information only for a subset of pixels per image and are thus
much faster to obtain than dense annotations. We propose a novel set of losses
that exploit measures from vector fields used in physics, i.e., divergence and
curl, to effectively supervise predictions without ground truth annotations.

![Motivation](./static/network.jpg)

Exemplary, we show in the following figure a comparision between dense annotations (left)
and partial annotations (right). The former contains ground truth annotations 
for all instances, i.e., *things*, and each pixel in the background, i.e., *stuff*.
In contrast, the latter annotations contain annotations for a few instances and 
some blob-like labels for the background.

![Annotations](./static/annotations.jpg)

# Requirements
- We assume that ```wget``` is installed on your machine
- We use ```conda``` to setup a virtual environment - in case ```conda``` is not installed on your machine follow the offical [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) (we recommend miniconda)

# Setup
The following script will setup a virutal environment denoted as *pspa*
```bash
./setup.sh
```

# Datasets
- We use the [PhenoBench](https://www.phenobench.org/) dataset
- However, as mentioned in the paper, we use a subset of images during training to ensure that each unique instance appears only once
- Consequently, we provide in ```phenobench_auxiliary/split.yaml``` the filenames of images used during training
- To download the dataset and organize it in the expected format run the following script
```bash 
cd ./phenobench_auxiliary
./get_phenobench.sh <full/path/to/download/PhenoBench>
cd ..
```


# Pretrained Models
- We provide the [pretrained model](https://uni-bonn.sciebo.de/s/7gDikkAynXM3avv) using ERFNet as network architecture

# Inference

Before you start the inference you need to specify the path to the dataset in the corresponding configuration file, e.g.:
```yaml
data:
  path: <full/path/to/download/PhenoBench/test>
```
Please change only ```<full/path/to/download/PhenoBench>``` according to your previously specified path to download PhenoBench but keep ```test``` as a directory at the very end.

Next, you can run the model in inference mode
```python
conda activate pspa
python predict.py --config ./config/config-phenobench.yaml  --export <path/to/export_dir> --ckpt <path/to/checkpoint/model.ckpt>
```
In the specified ```<path/to/export_dir>``` you will find the predicted semantics and plant instances.

In case you want to apply our proposed *merging procedure* you need run the following subsequently
```python
conda activate pspa
python auxiliary/merge_plants_sem.py --semantics <path/to/previously/predicted/semantics> --plants <path/to/previously/predicted/plant/instances> --export <path/to/merged_export_dir>
```

# Training
Similiarly you can train a new model
```python
conda activate pspa
python train.py --config ./config/config-phenobench.yaml --export <path/to/export_dir>
```

- Before you start the training you need to specify the path to the dataset in the corresponding configuration file, e.g.:
```yaml
data:
  path: <full/path/to/download/PhenoBench>
```
where ```<full/path/to/download/PhenoBench>``` matches the path you specified to download the PhenoBench dataset.

# License
This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary [here](https://creativecommons.org/licenses/by-nc/4.0/).
