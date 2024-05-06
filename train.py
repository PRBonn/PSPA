import argparse
import time
from pathlib import Path
from typing import Dict, List

import oyaml as yaml  # type: ignore
import pytorch_lightning as pl  # type: ignore
from pytorch_lightning import Trainer  # type: ignore
from pytorch_lightning.callbacks import Callback  # type: ignore
from pytorch_lightning.callbacks import (LearningRateMonitor,  # type: ignore
                                         ModelCheckpoint)

from callbacks import (ConfigCallback, PostprocessorCallback,
                       VisualizerCallback, get_postprocessors, get_visualizers)
from datasets import get_datamodule
from modules.module import Module
from modules.networks import get_network


def parse_args() -> Dict[str, str]:
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=Path, required=True, help='Path to configuration file (.yaml).')
  parser.add_argument('--export', type=Path, required=True, help='Path to export directory.')
  parser.add_argument('--ckpt', type=Path, required=False, default=None, help='Path to checkpoint file')
  parser.add_argument("--resume", required=False, action='store_true') # implies default = False

  args = vars(parser.parse_args())

  return args

def load_config(path: Path) -> Dict:
  with open(path) as f:
    cfg = yaml.safe_load(f)

  return cfg

def create_callbacks(cfg: Dict) -> List[Callback]:
  callbacks = []

  # ---- Fixed callbacks ----
  lr_monitor = LearningRateMonitor(logging_interval='epoch')
  callbacks.append(lr_monitor)

  fname_ckpt_01 = '{epoch:02d}_min_{train_loss:.4f}'
  ckpt_01 = ModelCheckpoint(monitor='train_loss', filename=fname_ckpt_01, mode='min', save_last=True, save_on_train_epoch_end=True)
  callbacks.append(ckpt_01)

  config_callback = ConfigCallback(cfg)
  callbacks.append(config_callback)

  # ---- Callbacks defined in config ----
  visualizer_callback = VisualizerCallback(get_visualizers(cfg))
  callbacks.append(visualizer_callback)

  postprocessor_callback = PostprocessorCallback(get_postprocessors(cfg))
  callbacks.append(postprocessor_callback)

  return callbacks

def main():
  args = parse_args()

  cfg = load_config(args['config']) # type: ignore
  if cfg.get('seed') is None:
    seed_val = int(time.time())
    cfg['seed'] = seed_val
  else:
    seed_val = cfg['seed'] 
  pl.utilities.seed.seed_everything(seed_val)

  datamodule = get_datamodule(cfg)

  network = get_network(cfg)

  module = Module(network, lr=cfg['train']['lr'], w_decay=cfg['train']['weight_decay'], warm_up_epochs=cfg['train']['warm_up_epochs'], steps=cfg['train']['steps'])
  if cfg['data']['cls_weights'] is not None:
    module.criterion_semantics.weights = cfg['data']['cls_weights']
  if (args['ckpt'] is not None) and (not args['resume']):
    module.load_model(args['ckpt']) # type: ignore

  # Setup trainer
  ckpt_path = None
  if args['resume']:
    ckpt_path = args['ckpt']
  trainer = Trainer(
      default_root_dir=args['export'],
      accelerator=cfg['train']['accelerator'],
      devices=cfg['train']['devices'],
      benchmark=cfg['train']['benchmark'],
      max_epochs=cfg['train']['max_epochs'],
      check_val_every_n_epoch=cfg['val']['check_val_every_n_epoch'],
      callbacks=create_callbacks(cfg),
      resume_from_checkpoint=ckpt_path) #, profiler="simple")

  trainer.fit(module, datamodule)

if __name__ == '__main__':
  main()
