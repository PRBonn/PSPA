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

# Setup
```bash
conda create -n pspa python=3.11
```

Please note that the cuda version depends on your local machine.

# Training

```python
python train.py <configs>
```

Before you start the training you need to specify the paths to the datasets in the corresponding configuration file.

# Testing
```python
python test.py <configs>
```
Before you start the testing you need to specify the paths to the datasets in the corresponding configuration file.

# Datasets
- We use the [PhenoBench](https://www.phenobench.org/) dataset

# Pretrained Models
- We provide the pretrained models using ERFNet as network architecture

# Notes

We are currently preparing the code that will be released by May 10th.

# License
This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary [here](https://creativecommons.org/licenses/by-nc/4.0/).
