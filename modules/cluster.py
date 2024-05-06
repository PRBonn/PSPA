import pdb
from abc import ABC, abstractmethod
from typing import Dict, Union

import torch
from cuml.cluster import DBSCAN


class ClusterModel(ABC):
  """ Generic representation of a cluster model.
  """
  @abstractmethod
  def cluster(self, embeddings: torch.Tensor, foreground: torch.Tensor) -> torch.Tensor:
    """ Perform the clustering for the task of instance segmentation.

    Args:
        embeddings (torch.Tensor): Spatial embeddings of shape [B x 2 x H x W]
        foreground (torch.Tensor): Foreground mask of shape [B x H x W]

    Returns:
        torch.Tensor: Results of instance segmentation of shape [B x H x W]
    """
    raise NotImplementedError
  
class DBSCANClusterModel(ClusterModel):
  def __init__(self, noise_label: int, eps: float=0.025, min_samples: int=64) -> None:
    self.noise_label = noise_label
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
      labels = self.model.fit_predict(emb[:, fg].T)
      labels = torch.Tensor(labels.get()).cuda() # this is not great: gpu (cupy) -> cpu -> gpu (torch)
      labels[labels >= 0] += 1 # avoid having an instance with id == 0
      labels[labels == -1] = self.noise_label

      instance_map[fg] = labels
      instance_maps.append(instance_map)

    return torch.stack(instance_maps)
  
def get_cluster_model(cfg: Dict) -> Union[ClusterModel, None]:
    
  name = cfg['cluster_model']['name']

  if name == 'dbscan':
    noise_label = cfg['cluster_model']['noise_label']
    eps = cfg['cluster_model']['eps']
    min_samples = cfg['cluster_model']['min_samples']
    
    return DBSCANClusterModel(noise_label, eps, min_samples)

