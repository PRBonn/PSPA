import pdb
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_lightning.callbacks import Callback  # type: ignore
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import distinctipy
import skimage.io

def hex_to_rgb(hex: str) -> torch.Tensor:
  hex = hex.strip('#')
  rgb = []
  for i in (0, 2, 4):
    decimal = int(hex[i:i+2], 16)
    rgb.append(decimal)
  
  return torch.Tensor([[rgb[0]], [rgb[1]], [rgb[2]]]).type(torch.uint8)

def random_hex_color() -> str:
  r = lambda: random.randint(0,255)
  hex_color = '#%02X%02X%02X' % (r(),r(),r())

  return hex_color

class Visualizer(ABC):
  """ Generic representation of a visualizer.
  """
  @abstractmethod
  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    raise NotImplementedError

  @abstractmethod
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    raise NotImplementedError

class InputImageVisualizer(Visualizer):
  """ Visualize the input images.
  """
  def __init__(self):
    self.name = 'input_images'
    self.tensor_to_pil = transforms.ToPILImage()

  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    log_dir = base_dir / self.name / f'{epoch:03d}'
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir

  @torch.no_grad()
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    for img, fname in zip(batch['img_unnormalized'], batch['fname']):
      img_pil = self.tensor_to_pil(img)
      img_pil.save(log_dir / fname)

class InstanceMapVisualizer(Visualizer):
  """ Visualize the results of instance segmentation.
  """
  def __init__(self,overlay: bool = False, alpha: float = 0.5):
    self.overlay = overlay
    self.alpha = alpha

    self.name = 'instance_segmentation'
    self.tensor_to_pil = transforms.ToPILImage()

  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    log_dir = base_dir / self.name / f'{epoch:03d}'
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir

  @torch.no_grad()
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    for img_unnormalized, fname, instance_map in zip(batch['img_unnormalized'], batch['fname'], outputs['instance_maps']):
      canvas = torch.zeros_like(img_unnormalized, dtype=torch.uint8)
      for instance_id in instance_map.unique():
        if instance_id == 0:
          continue

        mask = instance_map == instance_id
        color = hex_to_rgb(random_hex_color()).to(mask.device)
        canvas[:, mask] = color

      canvas = self.tensor_to_pil(canvas)
      if self.overlay:
        img = TF.to_pil_image(img_unnormalized)
        canvas = Image.blend(img, canvas, alpha=self.alpha)

      canvas.save(log_dir / fname)

class InstanceClusterVisualizer(Visualizer):
  """ Visualize the clusters of instance segmentation.
  """
  def __init__(self,overlay: bool = False, alpha: float = 0.5):
    self.name = 'instance_clusters'

  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    log_dir = base_dir / self.name / f'{epoch:03d}'
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir

  @torch.no_grad()
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    for fname, instance_map, embeddings in zip(batch['fname'], outputs['instance_maps'], outputs['embeddings']):
      instance_map = instance_map.cpu().numpy()
      emb_fg = embeddings[:, instance_map != 0].cpu().numpy()
      # emb_fg = embeddings.cpu().numpy()
      
      # colors_instances = distinctipy.get_colors(len(np.unique(instance_map)))
      # mymap = np.zeros((instance_map.shape[0], instance_map.shape[1], 3), dtype=np.uint8)
      # fig = plt.figure(frameon=False)
      # fig.set_size_inches(10.66, 10.66)
      # ax = plt.Axes(fig, [0., 0., 1., 1.])
      # ax.set_axis_off()
      # fig.add_axes(ax)
      # for idx, instance_id in enumerate(np.unique(instance_map)):
      #   if instance_id == 0:
      #     continue
      #   mask = instance_map == instance_id

      #   ax.scatter(emb_fg[0, mask], emb_fg[1, mask], color=colors_instances[idx], alpha=0.4, s=10.0)
      #   mymap[mask] = (int(np.round(colors_instances[idx][0]*255)), int(np.round(colors_instances[idx][1]*255)), int(np.round(colors_instances[idx][2]*255)))
                  
      # plt.gca().set_xlim(left=-2, right=2)
      # plt.gca().set_ylim(bottom=-2, top=2)
      # plt.gca().invert_yaxis()
      # plt.savefig(log_dir / fname, dpi=300)
      # plt.close('all')

      plt.scatter(emb_fg[0, :], emb_fg[1, :], c=instance_map[instance_map != 0], s=0.25, cmap='gist_ncar')
      ax=plt.gca() # get the axis
      ax.set_ylim(ax.get_ylim()[::-1])
      plt.savefig(log_dir / fname, dpi=300)
      plt.close('all')
      # skimage.io.imsave(log_dir / f'ins_{fname}', mymap)

class VectorFieldVisualizer(Visualizer):
  """ Visualize a vector field of offset vectors of vote per pixel.
  """
  def __init__(self) -> None:
    self.name = "vector_field"

  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    log_dir = base_dir / self.name / f'{epoch:03d}'
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir

  @torch.no_grad()
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    for coord_map, offsets, instances, fname in zip(outputs['cmap'], outputs['offsets'], batch['instances'], batch['fname']):
      coord_map = coord_map.cpu().numpy()
      offsets = offsets.cpu().numpy()
      instances = instances.cpu().numpy()
      for instance_id in np.unique(instances):
        mask = instances == instance_id
        if (instance_id == 0) or (instance_id == 255):
          offsets[:, mask] = 0
        offsets_copy = offsets.copy()
        offsets_copy[:,~mask] = 0

        plt.quiver(coord_map[0,::8,::8], coord_map[1, ::8, ::8],  offsets_copy  [0, ::8, ::8], offsets_copy [1, ::8, ::8], width=.001, angles='xy', scale_units='xy', scale=1, color=random_hex_color(), alpha=0.75)
      plt.savefig(log_dir / fname, dpi=900)
      plt.close('all')

class EmbeddingVisualizerGT(Visualizer):
  """ Visualize a plot of the embeddings using the ground truth instance annotations.
  """
  def __init__(self) -> None:
    self.name = 'embeddings_plot_gt'

  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    log_dir = base_dir / self.name / f'{epoch:03d}'
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir
  
  @torch.no_grad()
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    for embeddings, fname, instances in zip(outputs['embeddings'], batch['fname'], batch['instances']):
      for instance_id in torch.unique(instances):
        if instance_id == 0:
          continue
        mask = instances == instance_id
        emb_instance = embeddings[:, mask]

        x = emb_instance[0, :].detach().cpu().numpy()
        y = emb_instance[1, :].detach().cpu().numpy()
        plt.scatter(x, y, color=random_hex_color(), alpha=0.35, s=0.25)
        
      plt.savefig(log_dir / fname, dpi=300)
      plt.close('all')

class EmbeddingVisualizerFG(Visualizer):
  """ Visualize a plot of the embeddings belonging to foreground.
  """
  def __init__(self) -> None:
    self.name = 'embeddings_plot_fg'

  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    log_dir = base_dir / self.name / f'{epoch:03d}'
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir
  
  @torch.no_grad()
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    for embeddings, fname, foreground in zip(outputs['embeddings'], batch['fname'], outputs['foreground']):
        emb_instance = embeddings[:, foreground]

        x = emb_instance[0, :].detach().cpu().numpy()
        y = emb_instance[1, :].detach().cpu().numpy()
        # fig = plt.figure(frameon=False)
        # fig.set_size_inches(10.66, 10.66)
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        # ax.scatter(x, y, color='#0000ff', alpha=0.4, s=10.0)
        # plt.gca().set_xlim(left=-2, right=2)
        # plt.gca().set_ylim(bottom=-2, top=2)
        # plt.gca().invert_yaxis()
        # plt.savefig((log_dir / fname), dpi=600, format='png')
        # plt.close('all')
        plt.scatter(x, y, color='#0000ff', alpha=0.40, s=0.25)
        ax=plt.gca() # get the axis
        # ax.set_xlim(left=-1.6, right=1.6)
        # ax.set_ylim(bottom=-1.6, top=1.6)
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.savefig(log_dir / fname, dpi=300)
        plt.close('all')

class InstanceGTVisualizer(Visualizer):
  """ Visualize the ground truth instances. 
  """
  def __init__(self, overlay: bool = False, alpha: float = 0.5) -> None:
    self.overlay = overlay
    self.alpha = alpha

    self.name = 'instances_gt'
    self.tensor_to_pil = transforms.ToPILImage()

  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    log_dir = base_dir / self.name / f'{epoch:03d}'
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir

  @torch.no_grad()
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    for instances, fname, img_unnormalized in zip(batch['instances'], batch['fname'], batch['img_unnormalized']):
      h, w = instances.shape[:2]
      
      canvas = torch.zeros((3, h, w), dtype=torch.uint8, device=instances.device)
      for instance_id in torch.unique(instances):
        if instance_id == 0:
          continue
        mask = instances == instance_id
        if instance_id == 255:
          color = hex_to_rgb('#ffffff').to(mask.device)
        else:
          color = hex_to_rgb(random_hex_color()).to(mask.device) 
        canvas[:, mask] = color

      canvas = self.tensor_to_pil(canvas)
      if self.overlay:
        # img_fpath = Path(img_dir) / fname
        # img = Image.open(img_fpath)
        img = TF.to_pil_image(img_unnormalized)
        canvas = Image.blend(img, canvas, alpha=self.alpha)

      canvas.save(log_dir / fname)

class SemanticGTVisualizer(Visualizer):
  """ Visualize the semantic ground truths. 
  """
  def __init__(self, colors: List[str], overlay: bool = False, alpha: float = 0.5) -> None:
    self.colors = colors
    self.overlay = overlay
    self.alpha = alpha

    self.name = 'semantics_gt'
    self.tensor_to_pil = transforms.ToPILImage()

  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    log_dir = base_dir / self.name / f'{epoch:03d}'
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir

  @torch.no_grad()
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    for semantics, fname, img_unnormalized in zip(batch['semantics'], batch['fname'], batch['img_unnormalized']):
      h, w = semantics.shape[:2]
      
      canvas = torch.zeros((3, h, w), dtype=torch.uint8, device=semantics.device)
      for class_id in torch.unique(semantics):
        mask = semantics == class_id
        if class_id == 255:
          color = hex_to_rgb('#ffffff').to(mask.device)
        else:
          color = hex_to_rgb(self.colors[class_id]).to(mask.device)
        canvas[:, mask] = color

      canvas = self.tensor_to_pil(canvas)
      if self.overlay:
        # img_fpath = Path(img_dir) / fname
        # img = Image.open(img_fpath)
        img = TF.to_pil_image(img_unnormalized)
        canvas = Image.blend(img, canvas, alpha=self.alpha)

      canvas.save(log_dir / fname)

class SemanticPredVisualizer(Visualizer):
  """ Visualize the semantic predictions. 
  """
  def __init__(self, colors: List[str], overlay: bool = False, alpha: float = 0.5) -> None:
    self.colors = colors
    self.overlay = overlay
    self.alpha = alpha

    self.name = 'semantics_preds'
    self.tensor_to_pil = transforms.ToPILImage()

  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    log_dir = base_dir / self.name / f'{epoch:03d}'
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir

  @torch.no_grad()
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    for logits, fname, img_unnormalized in zip(outputs['logits'], batch['fname'], batch['img_unnormalized']):
      semantics = torch.argmax(logits, dim=0) 
      semantic_probs = F.softmax(logits, dim=0)

      h, w = semantics.shape[:2]
      
      canvas = torch.zeros((3, h, w), dtype=torch.uint8, device=semantics.device)
      for class_id in torch.unique(semantics):
        mask = semantics == class_id
        # prob_mask = semantic_probs[class_id, :]
        # mask[prob_mask < 0.90] = False
        color = hex_to_rgb(self.colors[class_id]).to(mask.device)
        canvas[:, mask] = color

      canvas = self.tensor_to_pil(canvas)
      if self.overlay:
        img = TF.to_pil_image(img_unnormalized)
        canvas = Image.blend(img, canvas, alpha=self.alpha)

      canvas.save(log_dir / fname)

def get_visualizers(cfg: Dict) -> List[Visualizer]:
  visualizers = []

  try:
    cfg['visualizers'].keys()
  except KeyError:
    return visualizers

  for name in cfg['visualizers'].keys():
    if name == 'input_image':
      vis = InputImageVisualizer()
      visualizers.append(vis)

    if name == 'instance_segmentation':
      overlay = cfg['visualizers'][name]['overlay']
      alpha = cfg['visualizers'][name]['alpha']
      vis = InstanceMapVisualizer(overlay, alpha)
      visualizers.append(vis)

    if name == 'instance_clusters':
      vis = InstanceClusterVisualizer()
      visualizers.append(vis)

    if name == 'embeddings_fg':
      vis = EmbeddingVisualizerFG()
      visualizers.append(vis)

    if name == 'embeddings_gt':
      vis = EmbeddingVisualizerGT()
      visualizers.append(vis)

    if name == 'instances_gt':
      overlay = cfg['visualizers'][name]['overlay']
      alpha = cfg['visualizers'][name]['alpha']
      vis = InstanceGTVisualizer(overlay, alpha)
      visualizers.append(vis)
    
    if name == 'semantics_pred':
      colors = cfg['visualizers'][name]['colors']
      overlay = cfg['visualizers'][name]['overlay']
      alpha = cfg['visualizers'][name]['alpha']
      vis = SemanticPredVisualizer(colors, overlay, alpha)
      visualizers.append(vis)

    if name == 'semantics_gt':
      colors = cfg['visualizers'][name]['colors']
      overlay = cfg['visualizers'][name]['overlay']
      alpha = cfg['visualizers'][name]['alpha']
      vis = SemanticGTVisualizer(colors, overlay, alpha)
      visualizers.append(vis)
      
    if name == 'vector_field':
      vis = VectorFieldVisualizer()
      visualizers.append(vis)

  return visualizers

class VisualizerCallback(Callback):
  def __init__(self, visualizers: List[Visualizer]):
    super().__init__()
    self.visualizers = visualizers

  # def on_train_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx):
  #   base_dir = Path(trainer.log_dir) / 'train' / 'visualize'
    
  #   for visualizer in self.visualizers:
  #     log_dir = visualizer.create_log_dir(base_dir, trainer.current_epoch)
  #     visualizer.process_batch(outputs, batch, log_dir)

  def on_validation_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    base_dir = Path(trainer.log_dir) / 'val' / 'visualize'
    
    for visualizer in self.visualizers:
      log_dir = visualizer.create_log_dir(base_dir, trainer.current_epoch)
      visualizer.process_batch(outputs, batch, log_dir)

  def on_test_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    base_dir = Path(trainer.log_dir) / 'test' / 'visualize'
    
    for visualizer in self.visualizers:
      log_dir = visualizer.create_log_dir(base_dir, trainer.current_epoch)
      visualizer.process_batch(outputs, batch, log_dir)

  def on_predict_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    base_dir = Path(trainer.log_dir) / 'predict' / 'visualize'
    
    for visualizer in self.visualizers:
      log_dir = visualizer.create_log_dir(base_dir, trainer.current_epoch)
      visualizer.process_batch(outputs, batch, log_dir)


