import argparse
from pathlib import Path
from typing import Dict, List

import oyaml as yaml  # type: ignore
from pytorch_lightning import Trainer  # type: ignore
from pytorch_lightning.callbacks import Callback  # type: ignore
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from callbacks import (ConfigCallback, ExporterCallback, PostprocessorCallback,
                       VisualizerCallback, get_exporter, get_postprocessors,
                       get_visualizers)
from datasets import get_datamodule
from modules.cluster import get_cluster_model
from modules.module import Module
from modules.networks import get_network


def parse_args() -> Dict[str, str]:
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=Path, required=True, help='Path to configuration file (.yaml).')
  parser.add_argument('--export', type=Path, required=True, help='Path to export directory.')
  parser.add_argument('--ckpt', type=Path, required=True, default=None, help='Path to checkpoint file')

  args = vars(parser.parse_args())

  return args

def load_config(path: Path) -> Dict:
  with open(path) as f:
    cfg = yaml.safe_load(f)

  return cfg

def create_callbacks(cfg: Dict) -> List[Callback]:
  callbacks = []

  # ---- Fixed callbacks ----
  config_callback = ConfigCallback(cfg)
  callbacks.append(config_callback)

  # ---- Callbacks defined in config ----
  visualizer_callback = VisualizerCallback(get_visualizers(cfg))
  callbacks.append(visualizer_callback)

  postprocessor_callback = PostprocessorCallback(get_postprocessors(cfg))
  callbacks.append(postprocessor_callback)

  exporter_callback = ExporterCallback(get_exporter(cfg))
  callbacks.append(exporter_callback)

  return callbacks

def main():
  args = parse_args()

  cfg = load_config(args['config']) # type: ignore

  datamodule = get_datamodule(cfg)

  network = get_network(cfg)

  module = Module(network, 
                  lr=cfg['train']['lr'], 
                  w_decay=cfg['train']['weight_decay'],
                  warm_up_epochs=cfg['train']['warm_up_epochs'],
                  cluster_model=get_cluster_model(cfg))
  
  module.load_model(args['ckpt']) # type: ignore

  # Setup trainer
  trainer = Trainer(
      default_root_dir=args['export'],
      accelerator=cfg['train']['accelerator'],
      devices=cfg['train']['devices'],
      benchmark=cfg['train']['benchmark'],
      callbacks=create_callbacks(cfg))

  trainer.predict(module, datamodule)

if __name__ == '__main__':
  main()
