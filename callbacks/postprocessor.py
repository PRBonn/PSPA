import pdb
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

# import cupy
import matplotlib.pyplot as plt
import numpy as np
import torch
from cuml.cluster import DBSCAN, HDBSCAN
from PIL import Image
from pytorch_lightning.callbacks import Callback  # type: ignore


class DBSCANModel():
  def __init__(self, eps: float=0.025, min_samples: int=64) -> None:
    self.eps = eps 
    self.min_samples = min_samples
    self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)

  def cluster(self, embeddings: torch.Tensor, foreground: torch.Tensor):
    batch_size = embeddings.shape[0]
    instance_maps = []

    for batch_idx in range(batch_size):
      fg = foreground[batch_idx] # [H x W]
      instance_map = torch.zeros_like(fg).float() # [H x W]
      if not torch.any(fg):
        instance_maps.append(instance_map)
        continue
      
      emb = embeddings[batch_idx] # [2 x H x W]
      labels = self.model.fit_predict(emb[:, fg].T) # , 'int64')

      labels = torch.Tensor(labels.get()).cuda() # this is not great: gpu (cupy) -> cpu -> gpu (torch)
      labels[labels >= 0] += 1 # avoid having an instance with id == 0

      instance_map[fg] = labels
      instance_maps.append(instance_map)
    
    return torch.stack(instance_maps)


class Postprocessor(ABC):
  """ Generic representation of a postprocessor.
  """
  @abstractmethod
  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    raise NotImplementedError

  @abstractmethod
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    raise NotImplementedError


class DBSCANPostprocessor(Postprocessor):
  def __init__(self, eps: float, min_samples: int):
    self.name = "dbscan"
    self.eps = eps 
    self.min_samples = min_samples
    self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)

  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    log_dir = base_dir / self.name / f'{epoch:03d}'
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir

  @torch.no_grad()
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    for emb, fg, fname, img_dir in zip(outputs['embeddings'], outputs['foreground'], batch['fname'], batch['img_dir']):
      if not torch.any(fg):
        continue
      cmap = outputs['cmap'][0]
      cmap_fg = cmap[:, fg].cpu().numpy()

      labels = self.model.fit_predict(emb[:, fg].T)#, 'int64')
      labels = torch.Tensor(labels.get()).numpy()
      labels[labels >= 0] += 1 # avoid having an instance with id == 0

      plt.scatter(cmap_fg[0, :], cmap_fg[1, :], c=labels*100, s=0.25, alpha=0.05, cmap='gist_ncar')
      img_fpath = Path(img_dir) / fname
      img = Image.open(img_fpath).transpose(Image.FLIP_TOP_BOTTOM)
      plt.imshow(img, extent=[-1,1,-1,1])
      
      ax=plt.gca() # get the axis
      ax.set_ylim(ax.get_ylim()[::-1])
      
      out_dir_instances = log_dir / 'instances'
      out_dir_instances.mkdir(exist_ok=True)
      plt.savefig(out_dir_instances / fname, dpi=300)
      plt.close('all')

      emb_fg = emb[:, fg].detach().cpu().numpy()
      plt.scatter(emb_fg[0, :], emb_fg[1, :], c=labels*100, s=0.25, cmap='gist_ncar')
      ax=plt.gca() # get the axis
      ax.set_ylim(ax.get_ylim()[::-1])
      
      out_dir_clusters = log_dir / 'clusters'
      out_dir_clusters.mkdir(exist_ok=True)
      plt.savefig(out_dir_clusters / fname, dpi=300)
      plt.close('all')

class HDBSCANPostprocessor(Postprocessor):
  def __init__(self, min_samples: int, min_cluster_size: int, cluster_selection_epsilon: float):
    self.name = "hdbscan"
    self.min_samples = min_samples
    self.min_cluster_size = min_cluster_size
    self.cluster_selection_epsilon = cluster_selection_epsilon
    self.model = HDBSCAN(min_samples=self.min_samples, min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon)

  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    log_dir = base_dir / self.name / f'{epoch:03d}'
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir

  @torch.no_grad()
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    for emb, fg, fname in zip(outputs['embeddings'], outputs['foreground'], batch['fname']):
      cmap = outputs['cmap'][0]
      labels = self.model.fit_predict(emb[:, fg].T, 'int64')

      cmap_fg = cmap[:, fg].cpu().numpy() 
      plt.scatter(cmap_fg[0, :], cmap_fg[1, :], c=np.array(labels.get()), s=0.25)
      out_dir_instances = log_dir / 'instances'
      out_dir_instances.mkdir(exist_ok=True)
      plt.savefig(out_dir_instances / fname, dpi=300)
      plt.close('all')
    
      emb_fg = emb[:, fg].detach().cpu().numpy()
      plt.scatter(emb_fg[0, :], emb_fg[1, :], c=np.array(labels.get()), s=0.25)
      out_dir_clusters = log_dir / 'clusters'
      out_dir_clusters.mkdir(exist_ok=True)
      plt.savefig(out_dir_clusters / fname, dpi=300)
      plt.close('all')

def get_postprocessors(cfg: Dict) -> List[Postprocessor]:
  postprocessors = []

  try:
    cfg['postprocessors'].keys()
  except KeyError:
    return postprocessors

  for name in cfg['postprocessors'].keys():
    if name == 'dbscan_postprocessor':
      eps = cfg['postprocessors'][name]['eps']
      min_samples = cfg['postprocessors'][name]['min_samples']
      postprocessor = DBSCANPostprocessor(eps=eps, min_samples=min_samples)
      postprocessors.append(postprocessor)

    if name == 'hdbscan_postprocessor':
      min_samples = cfg['postprocessors'][name]['min_samples']
      min_cluster_size = cfg['postprocessors'][name]['min_cluster_size']
      cluster_selection_epsilon = cfg['postprocessors'][name]['cluster_selection_epsilon']
      postprocessor = HDBSCANPostprocessor(min_samples=min_samples, min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon)
      postprocessors.append(postprocessor)

  return postprocessors

class PostprocessorCallback(Callback):
  def __init__(self, postprocessors: List[Postprocessor]):
    super().__init__()
    self.postprocessors = postprocessors

  def on_validation_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    base_dir = Path(trainer.log_dir) / 'val' / 'postprocess'
    
    for postprocessor in self.postprocessors:
      log_dir = postprocessor.create_log_dir(base_dir, trainer.current_epoch)
      postprocessor.process_batch(outputs, batch, log_dir)
  
  def on_test_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    base_dir = Path(trainer.log_dir) / 'test' / 'postprocess'
    
    for postprocessor in self.postprocessors:
      log_dir = postprocessor.create_log_dir(base_dir, trainer.current_epoch)
      postprocessor.process_batch(outputs, batch, log_dir)

  def on_predict_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    base_dir = Path(trainer.log_dir) / 'predict' / 'postprocess'
    
    for postprocessor in self.postprocessors:
      log_dir = postprocessor.create_log_dir(base_dir, trainer.current_epoch)
      postprocessor.process_batch(outputs, batch, log_dir)