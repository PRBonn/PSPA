import argparse
import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Set

import numpy as np
import skimage.io
from tqdm import tqdm


@dataclass
class PathInfos:
  fname: str
  sem_dir: Path
  plant_dir: Path
  export_dir: Path


def merge(path_info: PathInfos) -> str:
  semantics = skimage.io.imread(path_info.sem_dir / path_info.fname)
  plants = skimage.io.imread(path_info.plant_dir / path_info.fname)

  # --- Restrict the semantic segmentation ---
  semantics = semantics * (plants > 0)

  # --- Ensure that each instance belongs to unique semantic class ---
  for instance_id in np.unique(plants):
    if instance_id == 0:
      continue
    mask = plants == instance_id
    class_max_freq = np.bincount(semantics[mask].flatten()).argmax()
    semantics[mask] = class_max_freq

  skimage.io.imsave(path_info.export_dir / 'semantics' / path_info.fname, semantics, check_contrast=False)
  skimage.io.imsave(path_info.export_dir / 'plant_instances' / path_info.fname, plants, check_contrast=False)
  
  return path_info.fname


def parse_args() -> Dict[str, Path]:
  parser = argparse.ArgumentParser()
  parser.add_argument('--semantics', type=Path, required=True, help='Path to dir containing predicted semantics .')
  parser.add_argument('--plants', type=Path, required=True, help='Path to dir containing predicted plant instances.')
  parser.add_argument('--export', type=Path, required=True, help='Path to export directory')

  args = vars(parser.parse_args())

  (args['export'] / 'semantics').mkdir(parents=True, exist_ok=True)
  (args['export'] / 'plant_instances').mkdir(parents=True, exist_ok=True)

  return args


def get_png_fnames(path_to_dir: Path) -> Set[str]:
  return set([fname for fname in os.listdir(path_to_dir) if fname.endswith('.png')])


def main():
  args = parse_args()

  fnames = get_png_fnames(args['semantics'])
  path_infos = []
  for fname in tqdm(fnames):
    path_info = PathInfos(fname, args['semantics'], args['plants'], args['export'])
    path_infos.append(path_info)

  with Pool() as pool:
    results = pool.imap_unordered(merge, path_infos)
    for result in results:
      print(f"Processing: {result}")


if __name__ == '__main__':
  main()