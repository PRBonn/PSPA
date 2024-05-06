from .erfnet.erfnet_branched import BranchedERFNet
from typing import Dict

def get_network(cfg: Dict):
  name = cfg['network']['name']
  num_classes = cfg['network']['num_classes']
  dim_embeddings = cfg['network']['dim_embeddings']

  if name == "ERFNet":
    return BranchedERFNet(num_classes, dim_embeddings)

  assert False, "The specified network is not implemented"

