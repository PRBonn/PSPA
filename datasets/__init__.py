from typing import Dict

from .standard_dataset import StandardDataModule

def get_datamodule(cfg):
  name = cfg['data']['name']

  if name == 'standard':
    return StandardDataModule(cfg)

  assert False, "The specified dataset is not implemented."
