import pdb
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn

import modules.networks.erfnet.erfnet as erfnet


class BranchedERFNet(nn.Module):
  def __init__(self, num_classes: int, dim_embeddings: int):
    super().__init__()
    self.num_classes = num_classes
    self.dim_embeddings = dim_embeddings

    self.encoder = erfnet.Encoder(pretrained=False)

    self.decoders = nn.ModuleList()

    # decoder for semantic segmenation
    self.decoders.append(erfnet.Decoder(num_classes))

    # decoder for embeddings
    self.decoders.append(erfnet.Decoder(dim_embeddings))

    # self.load_encoder_decoder_pretrained()
    self.init_output()

  def init_output(self):
    with torch.no_grad():
      output_conv = self.decoders[1].output_conv
      output_conv.bias.fill_(0)
      output_conv.weight.fill_(0)

  def load_encoder_decoder_pretrained(self):
        path = "/home/janw/projects/instance_segmentation/pretrained/erfnet/erfnet_pretrained.pth"
        ckpt_dict = torch.load(path)
        
        ckpt_dict_encoder_new = {}
        ckpt_dict_encoder_new['state_dict'] = OrderedDict()
        ckpt_dict_decoder_new = {}
        ckpt_dict_decoder_new['state_dict'] = OrderedDict()

        for key in ckpt_dict.keys():
          if 'encoder' in key:
            key_new = key.replace("module.encoder.", "")
            ckpt_dict_encoder_new['state_dict'][key_new] = ckpt_dict[key].clone()
          if 'decoder' in key:
            if not 'output' in key:
              key_new = key.replace("module.decoder.", "")
              ckpt_dict_decoder_new['state_dict'][key_new] = ckpt_dict[key].clone()

        del ckpt_dict

        self.encoder.load_state_dict(ckpt_dict_encoder_new['state_dict'], strict=True)
        self.decoders[0].load_state_dict(ckpt_dict_decoder_new['state_dict'], strict=False)
        self.decoders[1].load_state_dict(ckpt_dict_decoder_new['state_dict'], strict=False)

  def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    output = self.encoder(input)

    semantic_branch_output = self.decoders[0].forward(output)
    embedding_branch_output = torch.tanh(self.decoders[1].forward(output))

    return semantic_branch_output, embedding_branch_output
