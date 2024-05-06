import os
from enum import Enum
from pathlib import Path
from typing import List


class Stage(Enum):
  TRAIN = 'train'
  VAL = 'val'
  TEST = 'test'
  PREDICT = 'predict'

def is_image(filename: str) -> bool:
  """ Check whether or not a given file is an image based on its format.

  Args:
      filename (str): filename

  Returns:
      bool: whether or not a given file is an image
  """
  return any(filename.endswith(ext) for ext in ['.jpg', '.png'])

def get_img_fnames_in_dir(path_to_dir: Path) -> List[str]:
  """ Get the filenames of all images in a certain directory.

  Args:
      path_to_dir (Path): path to directory

  Returns:
      List[str]: contain the full path to all images in given directory
  """
  fnames = []

  files_in_dir = os.listdir(path_to_dir)
  for fname in files_in_dir:
    if is_image(fname):
      fnames.append(fname)

  return fnames
