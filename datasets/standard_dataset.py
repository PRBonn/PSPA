import math
import pdb
import random
import time
import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import skimage.io
import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets.img_normalizer import ImageNormalizer, get_image_normalizer
from datasets.utils import Stage, get_img_fnames_in_dir

# props to: https://github.com/huggingface/pytorch-image-models/pull/140/commits/a7f570c9b72369fd75e15733d5645d09039d5f9e
#  - discussion: https://github.com/huggingface/pytorch-image-models/pull/140
class _RepeatSampler():
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class StandardDataModule(pl.LightningDataModule):
  """ Contains all the steps necssary to process standard datasets.
  """

  def __init__(self, cfg: Dict):
    super().__init__()
    self.cfg = cfg

  def get_predict_fnames(self) -> List[str]:
    image_dir = Path(self.cfg['data']['path']) / 'images'
    fnames = get_img_fnames_in_dir(image_dir)

    return fnames

  def setup(self, stage: Optional[str] = None) -> None:
    """ Operations to perform on every GPU.

    Args:
        stage (Optional[str], optional): Separate logic for trainer.{fit,validate,test,predict}. Defaults to None.
    """
    dataset_dir = Path(self.cfg['data']['path'])
    img_normalizer = get_image_normalizer(self.cfg)

    if stage != 'predict':
      path_split_file = dataset_dir / 'split.yaml'
      with path_split_file.open() as istream:
        split_info = yaml.safe_load(istream)

      # ------ Create Datasets for Training ------
      train_fnames = sorted(split_info['train'])
      if self.cfg['train']['size']:
        train_fnames = train_fnames[:self.cfg['train']['size']]
      self.ds_train = StandardDataset(dataset_dir, train_fnames, stage=Stage.TRAIN, img_normalizer=img_normalizer)

      # ------ Create Datasets for Validation ------
      val_fnames = sorted(split_info['val'])
      if self.cfg['val']['size']:
        val_fnames = val_fnames[:self.cfg['val']['size']]
      self.ds_val = StandardDataset(dataset_dir, val_fnames, stage=Stage.VAL, img_normalizer=img_normalizer)

      # ------ Create Datasets for Testing ------
      test_fnames = sorted(split_info['test'])
      if self.cfg['test']['size']:
        test_fnames = test_fnames[:self.cfg['test']['size']]
      self.ds_test = StandardDataset(dataset_dir, test_fnames, stage=Stage.TEST, img_normalizer=img_normalizer)

    else:
      # ------ Create Datasets for Inference ------
      predict_fnames = sorted(self.get_predict_fnames())
      if self.cfg['predict']['size']:
        predict_fnames = predict_fnames[:self.cfg['predict']['size']]
      self.ds_predict = StandardDataset(dataset_dir, predict_fnames, stage=Stage.PREDICT, img_normalizer=img_normalizer)

  def train_dataloader(self) -> DataLoader:
      # Return DataLoader for Training Data
      shuffle: bool = self.cfg['train']['shuffle']
      batch_size: int = self.cfg['train']['batch_size']
      n_workers: int = self.cfg['data']['num_workers']

      persistent_workers = False
      if n_workers > 0:
        persistent_workers = True
      loader = MultiEpochsDataLoader(self.ds_train, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers, drop_last=True, pin_memory=True, persistent_workers=persistent_workers)
      # MultiEpochsDataLoader
      return loader

  def val_dataloader(self) -> DataLoader:
      # Return DataLoader for Validation Data
      batch_size: int = self.cfg['val']['batch_size']
      n_workers: int = self.cfg['data']['num_workers']
      persistent_workers = False
      if n_workers > 0:
        persistent_workers = True
      loader = DataLoader(self.ds_val, batch_size=batch_size, shuffle=False, num_workers=n_workers, drop_last=True, pin_memory=True, persistent_workers=persistent_workers)

      return loader

  def test_dataloader(self) -> DataLoader:
      # Return DataLoader for Testing Data
      batch_size: int = self.cfg['test']['batch_size']
      n_workers: int = self.cfg['data']['num_workers']

      loader = DataLoader(self.ds_test, batch_size=batch_size, shuffle=False, num_workers=n_workers, drop_last=True, pin_memory=True, persistent_workers=False)

      return loader

  def predict_dataloader(self):
    batch_size: int = self.cfg['predict']['batch_size']
    n_workers: int = self.cfg['data']['num_workers']

    loader = DataLoader(self.ds_predict, batch_size=batch_size, num_workers=n_workers, shuffle=False, pin_memory=True, persistent_workers=False)

    return loader

class StandardDataset(Dataset):
  """ This class represents our standard dataset which has the following structure:
  
  ├── images
  │   └── 00000_P0030684.png
      └── ...
  ├── plant_instances
  │   └── 00000_P0030684.png
      └── ...
  ├── semantics
  │   └── 00000_P0030684.png
      └── ...
  └── split.yaml
  """

  def __init__(self, dataset_dir: Path, filenames: List[str], stage: Stage, img_normalizer: ImageNormalizer) -> None:
    """ Constructor of our standard dataset

    Args:
        dataset_dir (Path): Path to directory that contains the dataset
        filenames (List[str]): List of filenames that should be considered
        stage (Stage): Define the stage of the model {train, val, test, predict}
        img_normalizer (ImageNormalizer): Specific image normalizer
    """
    super().__init__()

    assert dataset_dir.exists(), f"The path to the dataset directory {dataset_dir} does not exists."
    self.dataset_dir = dataset_dir
    
    self.filenames = filenames
    
    self.stage = stage
    
    self.img_normalizer = img_normalizer

    self.image_dir = self.dataset_dir / 'images'
    self.semantic_dir = self.dataset_dir / 'semantics'
    self.instance_dir = self.dataset_dir / 'plant_instances'

    self.img_to_tensor = transforms.ToTensor()

  def imread_anno(self, directory: Path, fname: str) -> torch.Tensor:
    fpath = directory / fname

    anno = skimage.io.imread(fpath)
    anno = torch.Tensor(anno.astype(np.int64)).type(torch.int64)

    return anno

  def __getitem__(self, index: int) -> Dict:
    # ------ Load image from disk and process ------
    image_fpath = self.image_dir / self.filenames[index]
    image_pil = Image.open(image_fpath).convert('RGB')
    img_unnormalized = self.img_to_tensor(image_pil)  # torch.float32 and [C x H x W] 
    img_unnormalized = TF.resize(img_unnormalized, size=[1024, 1024], interpolation=InterpolationMode.BILINEAR)
    
    if self.stage == Stage.PREDICT:
      return {'image': self.img_normalizer(img_unnormalized), 
              'img_unnormalized': img_unnormalized, 
              'fname': self.filenames[index], 
              'img_dir': str(self.image_dir)}

    # ------ Load semantics from disk ------
    semantics = self.imread_anno(self.semantic_dir, self.filenames[index]).squeeze() # torch.int64 and [H x W]
    m1 = semantics == 3
    semantics[m1] = 1
    m2 = semantics == 4 
    semantics[m2] = 2
    semantics = TF.resize(semantics.unsqueeze(0), size=[1024, 1024], interpolation=InterpolationMode.NEAREST).squeeze()
    
    # ------ Load instances from disk ------
    instances = self.imread_anno(self.instance_dir, self.filenames[index]).squeeze() # torch.int64 and [H x W]
    instances = TF.resize(instances.unsqueeze(0), size=[1024, 1024], interpolation=InterpolationMode.NEAREST).squeeze()

    if self.stage == Stage.TRAIN:
      valid_instance_mask = (instances != 0) & (instances != 255)
      if (random.random() > 0.50) and (valid_instance_mask.sum() > 0):
        rand_instance_idx = random.choice(list(instances[valid_instance_mask].unique()))
        mask_instance = (instances == rand_instance_idx)
        rows, cols = torch.where(mask_instance > 0)
        center_y, center_x = int(rows.float().mean().round()), int(cols.float().mean().round())
        CROP_HEIGHT = random.randint(896, 960)
        CROP_WIDTH = CROP_HEIGHT
        while True:
          start_x = random.randint(0, 1024 - CROP_WIDTH)
          end_x = start_x + CROP_WIDTH
          start_y = random.randint(0, 1024 - CROP_HEIGHT)
          end_y = start_y + CROP_HEIGHT
          if (start_x <= center_x <= end_x) and (start_y <= center_y <= end_y):
            break
        # while True:
        #   start_y = random.randint(0, center_y)
        #   end_y = start_y + CROP_HEIGHT
        #   if (end_y > center_y) and (end_y < 1024):
        #     break 
        # while True:
        #   start_x = random.randint(0, center_x)
        #   end_x = start_x + CROP_WIDTH
        #   if (end_x > center_x) and (end_x < 1024):
        #     break
        img_unnormalized = img_unnormalized[:, start_y:end_y, start_x:end_x].clone()
        semantics = semantics[start_y:end_y, start_x:end_x].clone()
        instances = instances[start_y:end_y, start_x:end_x].clone()
        img_unnormalized = TF.resize(img_unnormalized, size=[1024, 1024], interpolation=InterpolationMode.BILINEAR)
        semantics = TF.resize(semantics.unsqueeze(0), size=[1024, 1024], interpolation=InterpolationMode.NEAREST).squeeze()
        instances = TF.resize(instances.unsqueeze(0), size=[1024, 1024], interpolation=InterpolationMode.NEAREST).squeeze()
    
    if self.stage == Stage.TRAIN:
      if random.random() > 0.50: # 0.10
        valid_instance_mask = (instances != 0) & (instances != 255)
        valid_instance_ids = list(torch.unique(instances[valid_instance_mask])) 
        if len(valid_instance_ids) > 2:
          rand_idx = random.choice(valid_instance_ids)
          mask_rand_idx = instances == rand_idx
          instances[mask_rand_idx] = 255
          semantics[mask_rand_idx] = 255
      if random.random() > 0.50: # 0.10
        valid_instance_mask = (instances != 0) & (instances != 255)
        valid_instance_ids = list(torch.unique(instances[valid_instance_mask])) 
        if len(valid_instance_ids) > 2:
          rand_idx = random.choice(valid_instance_ids)
          mask_rand_idx = instances == rand_idx
          instances[mask_rand_idx] = 255
          semantics[mask_rand_idx] = 255
    
    semantics = semantics.squeeze(0) # [H x W]
    instances = instances.squeeze(0) # [H x W]
    
    if self.stage == Stage.TRAIN:
      if random.random() > 0.25:
        angle = random.choice([90, 270])
        img_unnormalized = TF.rotate(img_unnormalized, angle=angle, interpolation=InterpolationMode.BILINEAR)
        semantics = TF.rotate(semantics.unsqueeze(0), angle=angle, interpolation=InterpolationMode.NEAREST).squeeze(0)
        instances = TF.rotate(instances.unsqueeze(0), angle=angle, interpolation=InterpolationMode.NEAREST).squeeze(0)
      if random.random() > 0.25:
        img_unnormalized = torch.flip(img_unnormalized, [1])
        semantics = torch.flip(semantics, [0])
        instances = torch.flip(instances, [0])
      if random.random() > 0.25:
        img_unnormalized = torch.flip(img_unnormalized, [2])
        semantics = torch.flip(semantics, [1])
        instances = torch.flip(instances, [1])

    img = self.img_normalizer(img_unnormalized)

    img_aug_90 = img_unnormalized.clone()
    img_aug_180 = img_unnormalized.clone()
    img_aug_270 = img_unnormalized.clone()
    if self.stage == Stage.TRAIN:
      if random.random() > 0.5:
        img_aug_90 = TF.adjust_brightness(img_aug_90, brightness_factor=random.uniform(0.8, 1.2))
        img_aug_180 = TF.adjust_brightness(img_aug_180, brightness_factor=random.uniform(0.8, 1.2))
        img_aug_270 = TF.adjust_brightness(img_aug_270, brightness_factor=random.uniform(0.8, 1.2))
        
      if random.random() > 0.5:
        img_aug_90 = TF.adjust_contrast(img_aug_90, contrast_factor=random.uniform(0.8, 1.2))
        img_aug_180 = TF.adjust_contrast(img_aug_180, contrast_factor=random.uniform(0.8, 1.2))
        img_aug_270 = TF.adjust_contrast(img_aug_270, contrast_factor=random.uniform(0.8, 1.2))
        
      if random.random() > 0.5:
        img_aug_90 = TF.adjust_saturation(img_aug_90, saturation_factor=random.uniform(0.8, 1.2))
        img_aug_180 = TF.adjust_saturation(img_aug_180, saturation_factor=random.uniform(0.8, 1.2))
        img_aug_270 = TF.adjust_saturation(img_aug_270, saturation_factor=random.uniform(0.8, 1.2))
        
      if random.random() > 0.5:
        img_aug_90 = TF.adjust_hue(img_aug_90, hue_factor=random.uniform(-0.0125, 0.0125))
        img_aug_180 = TF.adjust_hue(img_aug_180, hue_factor=random.uniform(-0.0125, 0.0125))
        img_aug_270 = TF.adjust_hue(img_aug_270, hue_factor=random.uniform(-0.0125, 0.0125))

      if random.random() > 0.5:
        img_aug_90 = torch.clip(img_aug_90 + (torch.randn_like(img_aug_90) * 0.01), 0.0, 1.0)
        img_aug_180 = torch.clip(img_aug_180 + (torch.randn_like(img_aug_180) * 0.01), 0.0, 1.0)
        img_aug_270 = torch.clip(img_aug_270 + (torch.randn_like(img_aug_270) * 0.01), 0.0, 1.0)

    img_rot_90 = TF.rotate(img_aug_90, angle=90, interpolation=InterpolationMode.BILINEAR)
    img_rot_90 = self.img_normalizer(img_rot_90)

    img_rot_180 = TF.rotate(img_aug_180, angle=180, interpolation=InterpolationMode.BILINEAR)
    img_rot_180 = self.img_normalizer(img_rot_180)

    img_rot_270 = TF.rotate(img_aug_270, angle=270, interpolation=InterpolationMode.BILINEAR)
    img_rot_270 = self.img_normalizer(img_rot_270)

    return {'img_unnormalized': img_unnormalized, 
            'image': img, 
            'img_rot_90': img_rot_90,
            'img_rot_180': img_rot_180, 
            'img_rot_270': img_rot_270, 
            'semantics': semantics, 
            'instances': instances, 
            'fname': self.filenames[index], 
            'img_dir': str(self.image_dir)}

  def __len__(self) -> int:
    return len(self.filenames)