from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import skimage.io  # type: ignore
import torch
from pytorch_lightning.callbacks import Callback  # type: ignore


class Exporter(ABC):
  """ Generic representation of an exporter.
  """
  @abstractmethod
  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    raise NotImplementedError

  @abstractmethod
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    raise NotImplementedError
  
class SemanticExporter(Exporter):
  def __init__(self):
    self.name = 'semantics'

  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    log_dir = base_dir / self.name / f'{epoch:03d}'
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir
  
  @torch.no_grad()
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    for fname, semantics in zip(batch['fname'], outputs['semantics']):
      semantics = semantics.cpu().numpy().astype(np.uint8)

      fpath = log_dir / fname
      skimage.io.imsave(fpath, semantics, check_contrast=False)

class InstanceExporter(Exporter):
  def __init__(self):
    self.name = 'instances'

  def create_log_dir(self, base_dir: Path, epoch: int) -> Path:
    log_dir = base_dir / self.name / f'{epoch:03d}'
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir
  
  @torch.no_grad()
  def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any], log_dir: Path) -> None:
    for fname, instance_map in zip(batch['fname'], outputs['instance_maps']):
      instance_map = instance_map.cpu().numpy().astype(np.uint8)

      fpath = log_dir / fname
      skimage.io.imsave(fpath, instance_map, check_contrast=False)

def get_exporter(cfg: Dict) -> List[Exporter]:
  exporters = []

  try:
    cfg['exporters'].keys()
  except KeyError:
    return exporters

  for name in cfg['exporters'].keys():
    if name == 'semantics':
      exporter = SemanticExporter()
      exporters.append(exporter)

    if name == 'instances':
      exporter = InstanceExporter()
      exporters.append(exporter)

  return exporters

class ExporterCallback(Callback):
  def __init__(self, exporters: List[Exporter]):
    super().__init__()
    self.exporters = exporters

  def on_train_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    base_dir = Path(trainer.log_dir) / 'train' / 'export'
    
    for exporter in self.exporters:
      log_dir = exporter.create_log_dir(base_dir, trainer.current_epoch)
      exporter.process_batch(outputs, batch, log_dir)

  def on_validation_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    base_dir = Path(trainer.log_dir) / 'val' / 'export'
    
    for exporter in self.exporters:
      log_dir = exporter.create_log_dir(base_dir, trainer.current_epoch)
      exporter.process_batch(outputs, batch, log_dir)
  
  def on_test_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    base_dir = Path(trainer.log_dir) / 'test' / 'export'
    
    for exporter in self.exporters:
      log_dir = exporter.create_log_dir(base_dir, trainer.current_epoch)
      exporter.process_batch(outputs, batch, log_dir)

  def on_predict_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    base_dir = Path(trainer.log_dir) / 'predict' / 'export'
    
    for exporter in self.exporters:
      log_dir = exporter.create_log_dir(base_dir, trainer.current_epoch)
      exporter.process_batch(outputs, batch, log_dir)