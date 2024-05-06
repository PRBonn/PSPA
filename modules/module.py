import math
import pdb
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pytorch_lightning as pl  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torchvision.transforms.functional as TF
from kornia import filters, morphology
from torchvision.transforms import InterpolationMode

from modules.cluster import ClusterModel
from modules.criterions.lovasz import IoULovaszLoss, lovasz_softmax
from modules.criterions.xentropy import CrossEntropy
from modules.evaluation.panoptic_quality import PanopticEvaluator
from modules.utils import Discrete2DVectorFieldAnalyst, GradientFilter

IMG_WIDTH = 1024
IMG_HEIGHT = 1024

def apply_offsets(xym: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
  """_summary_

  Args:
      xym (torch.Tensor): _description_ ... of shape [B x 2 x H x W]
      offsets (torch.Tensor): _description_ ... of shape [B x 2 x H x W]

  Returns:
      torch.Tensor: _description_
  """

  x_col = torch.round((((IMG_WIDTH - 1) * xym[:,0]) + (IMG_WIDTH - 1)) / 2) # [B x H x W]
  x_col = x_col.clip(0, IMG_WIDTH - 1).long()

  y_row = torch.round((((IMG_HEIGHT - 1) * xym[:,1]) + (IMG_HEIGHT - 1)) / 2) # [B x H x W]
  y_row = y_row.clip(0, IMG_HEIGHT - 1).long()

  x_offsets = offsets[:, 0, y_row, x_col]
  y_offsets = offsets[:, 1, y_row, x_col]

  xym_new = xym.clone()
  xym_new[:, 0] = xym_new[:,0] + x_offsets
  xym_new[:, 1] = xym_new[:,1] + y_offsets

  return xym_new

def cauchy_loss(residuals: torch.Tensor, c: float) -> torch.Tensor:
  losses = torch.log(0.5 * ((residuals / c)**2) + 1)
  return losses

def geman_mc_clure_loss(residuals: torch.Tensor, c: float) -> torch.Tensor:
  auxiliary = (residuals / c)**2
  losses = (2 * auxiliary) / (auxiliary + 4)

  return losses 
  
class Module(pl.LightningModule):
  def __init__(self,
               network: nn.Module, 
               lr: float, 
               w_decay: float, 
               warm_up_epochs: int, 
               cluster_model: Optional[ClusterModel]=None,
               steps: Optional[Dict[str,bool]]=None):
    super().__init__()

    self.network = network
    var = - (0.03**2) / (2 * math.log(0.90)) # 
    self.criterion_iou_lovasz = IoULovaszLoss(variance=var)
    self.criterion_semantics = CrossEntropy()

    self.save_hyperparameters('lr', 'w_decay', 'warm_up_epochs')

    xm = torch.linspace(-1, 1, IMG_WIDTH).view(1, 1, -1).expand(1, IMG_HEIGHT, IMG_WIDTH)  # x-coords in rows
    ym = torch.linspace(-1, 1, IMG_HEIGHT).view(1, -1, 1).expand(1, IMG_HEIGHT, IMG_WIDTH) # y-coords in cols
    xym = torch.cat((xm, ym), 0).unsqueeze(0) # [1 x 2 x H x W]
    self.register_buffer('xym', xym)

    x_step_size = float(xm[0,0,1] - xm[0,0,0])
    y_step_size = float(ym[0,1,0] - ym[0,0,0])
    self.vector_field_analyst = Discrete2DVectorFieldAnalyst(x_step_size, y_step_size, gradient_filter=GradientFilter(kernel_size='3x3'))

    # evaluation metrics semantic segmentation
    self.metric_train_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=self.network.num_classes, ignore_index=255)
    self.metric_val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=self.network.num_classes, ignore_index=255)
    self.metric_test_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=self.network.num_classes, ignore_index=255)

    # evaluation metrics panoptic quality
    self.metric_pq = PanopticEvaluator()

    self.cluster_model = cluster_model
    self.steps = steps

    self.dilation_kernel = torch.ones(5,5).to(torch.device('cuda:0'))

  def load_model(self, ckpt_path: Path) -> None:
    ckpt_dict = torch.load(ckpt_path)
    self.load_state_dict(ckpt_dict['state_dict'])
    print("{} Loaded pretrained weights".format(7*'#'))

  def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Forward pass of network

    Args:
        img (torch.Tensor): input rgb image of shape [B x C x H x W]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
          - semantics of shape [B x num_classes x H x W]
          - embeddings of shape [B x dim_embeddings x H x W]
    """
    return self.network.forward(img)
  
  def on_train_start(self):
    print("Training is started!")
    print("Steps are: ", self.steps)

  def training_step(self, batch: dict, batch_idx: int):
    batch_size = batch['image'].shape[0]
    epoch = float(self.trainer.current_epoch) # float required by logger
    if self.xym.device != self.device:
      self.xym = self.xym.to(self.device)

    logits, offsets = self.forward(batch['image'])

    # --------- process semantic predictions ---------
    with torch.no_grad():
      semantic_preds = torch.argmax(logits, dim=1) # [B x H x W]
      self.metric_train_iou(semantic_preds, batch['semantics'])
      fg = semantic_preds > 0

      gt_fg = (batch['semantics'] != 0) & (batch['semantics'] != 255)
      fg[gt_fg] = True

    loss_semantics = self.criterion_semantics(logits, batch['semantics'], mask_keep=batch['semantics'] != 255)
    self.log_dict({'avg_semantic_loss': loss_semantics, 'step': epoch}, on_epoch=True, on_step=False, batch_size=batch_size)
    loss = loss_semantics

    # --------- process embeddings ---------
    img_height, img_width = batch['image'].shape[2:]
    embeddings = self.xym[:,:,:img_height, :img_width].contiguous() + offsets # [B x 2 x H x W]
    
    avg_iou_loss = self.criterion_iou_lovasz.forward(batch['instances'], embeddings, foreground=None)
    self.log_dict({'avg_iou_loss': avg_iou_loss, 'step': epoch}, on_epoch=True, on_step=False, batch_size=batch_size)

    if self.steps['iou']:
      if not avg_iou_loss.isnan():
        loss += 5.0 * avg_iou_loss
    
    w_emb = 1.0 
    if self.trainer.current_epoch < 128:
       w_emb = (self.trainer.current_epoch / 128) ** 2.0

    with torch.no_grad():
      mask_background = (~fg).unsqueeze(1) # [B x 1 x H x W]
      mask_background = morphology.dilation(mask_background, kernel=self.dilation_kernel)
      mask_background = mask_background.bool()

    dxdx = self.vector_field_analyst.compute_dxdx(offsets) # [B x 1 x H x W]
    dydy = self.vector_field_analyst.compute_dydy(offsets)

    dxdy = self.vector_field_analyst.compute_dxdy(offsets)
    dydx = self.vector_field_analyst.compute_dydx(offsets)

    dxdx_fg = dxdx[~mask_background]
    dydy_fg = dydy[~mask_background]
    div_fg = dxdx_fg + dydy_fg

    dxdy_fg = dxdy[~mask_background]
    dydx_fg = dydx[~mask_background]
    curl_fg = dydx_fg - dxdy_fg

    c = 1.0

    if self.steps['divergence']:
      div_residuals = div_fg - (torch.ones_like(div_fg) * (-2))
      div_loss = geman_mc_clure_loss(div_residuals, c=c).mean()
      self.log_dict({'div_loss': div_loss, 'step': epoch}, on_epoch=True, on_step=False, batch_size=batch_size)
      if not div_loss.isnan():
        loss += w_emb * (1.0 * div_loss)

      res_01 = (dxdx_fg - dydy_fg) - (torch.zeros_like(dxdx_fg))
      res_01_loss = geman_mc_clure_loss(res_01, c=c).mean()
      self.log_dict({'res_01_loss': res_01_loss, 'step': epoch}, on_epoch=True, on_step=False, batch_size=batch_size)
      if not res_01_loss.isnan():
        loss += w_emb * (1.0 * res_01_loss)

    if self.steps['curl']:
      curl_residuals = curl_fg - (torch.zeros_like(curl_fg))
      curl_loss = geman_mc_clure_loss(curl_residuals, c=c).mean()
      self.log_dict({'curl_loss': curl_loss, 'step': epoch}, on_epoch=True, on_step=False, batch_size=batch_size)
      if not curl_loss.isnan():
        loss += w_emb * (1.0 * curl_loss)

      res_02 = (dydx_fg + dxdy_fg) - (torch.zeros_like(dxdy_fg))
      res_02_loss = geman_mc_clure_loss(res_02, c=c).mean()
      self.log_dict({'res_02_loss': res_02_loss, 'step': epoch}, on_epoch=True, on_step=False, batch_size=batch_size)
      if not res_02_loss.isnan():
        loss += w_emb * (1.0 * res_02_loss)

    if self.steps['consistency']:
      rotations = random.sample(['90', '180', '270'], k=2)
      # ----------------------------- 90 -----------------------------
      if '90' in rotations:
        logits_rot_90, offsets_rot_90 = self.forward(batch['img_rot_90'])
        logits_rot_90 = TF.rotate(logits_rot_90, angle=-90, interpolation=InterpolationMode.BILINEAR)
        offsets_rot_90 = TF.rotate(offsets_rot_90, angle=-90, interpolation=InterpolationMode.BILINEAR)
        
        loss_aux_x_90 = F.mse_loss(input=offsets_rot_90[:,0][fg], target=offsets[:,1][fg])
        self.log_dict({'loss_aux_x_90': loss_aux_x_90, 'step': epoch}, on_epoch=True, on_step=False, batch_size=batch_size)
        loss_aux_y_90 = F.mse_loss(input=-offsets_rot_90[:,1][fg], target=offsets[:,0][fg])
        self.log_dict({'loss_aux_y_90': loss_aux_y_90, 'step': epoch}, on_epoch=True, on_step=False, batch_size=batch_size)
        if not (loss_aux_x_90.isnan() and loss_aux_y_90.isnan()):
          loss += w_emb * (10.0 * (loss_aux_x_90 + loss_aux_y_90))

      # ----------------------------- 180 -----------------------------
      if '180' in rotations:
        logits_rot_180, offsets_rot_180 = self.forward(batch['img_rot_180'])
        logits_rot_180 = logits_rot_180.flip(2).flip(3)
        offsets_rot_180 = offsets_rot_180.flip(2).flip(3)
        
        loss_aux_x_180 = F.mse_loss(input=-offsets_rot_180[:,0][fg], target=offsets[:,0][fg])
        self.log_dict({'loss_aux_x_180': loss_aux_x_180, 'step': epoch}, on_epoch=True, on_step=False, batch_size=batch_size)
        loss_aux_y_180 = F.mse_loss(input=-offsets_rot_180[:,1][fg], target=offsets[:,1][fg])
        self.log_dict({'loss_aux_y_180': loss_aux_y_180, 'step': epoch}, on_epoch=True, on_step=False, batch_size=batch_size)
        if not (loss_aux_x_180.isnan() and loss_aux_y_180.isnan()):
          loss += w_emb * (10.0 * (loss_aux_x_180 + loss_aux_y_180))

      # ----------------------------- 270 -----------------------------
      if '270' in rotations:
        logits_rot_270, offsets_rot_270 = self.forward(batch['img_rot_270'])
        logits_rot_270 = TF.rotate(logits_rot_270, angle=-270, interpolation=InterpolationMode.BILINEAR)
        offsets_rot_270 = TF.rotate(offsets_rot_270, angle=-270, interpolation=InterpolationMode.BILINEAR)

        loss_aux_x_270 = F.mse_loss(input=-offsets_rot_270[:,0][fg], target=offsets[:,1][fg])
        self.log_dict({'loss_aux_x_270': loss_aux_x_270, 'step': epoch}, on_epoch=True, on_step=False, batch_size=batch_size)
        loss_aux_y_270 = F.mse_loss(input=offsets_rot_270[:,1][fg], target=offsets[:,0][fg])
        self.log_dict({'loss_aux_y_270': loss_aux_y_270, 'step': epoch}, on_epoch=True, on_step=False, batch_size=batch_size)
        if not (loss_aux_x_270.isnan() and loss_aux_y_270.isnan()):
          loss += w_emb * (10.0 * (loss_aux_x_270 + loss_aux_y_270))
    
      p_clean = F.softmax(logits, dim=1)
      p_augs = []
      if 'logits_rot_90' in locals().keys():
        p_aug_90 = F.softmax(logits_rot_90, dim=1)
        p_augs.append(p_aug_90)
      if 'logits_rot_180' in locals().keys():
        p_aug_180 = F.softmax(logits_rot_180, dim=1)
        p_augs.append(p_aug_180)
      if 'logits_rot_270' in locals().keys():
        p_aug_270 = F.softmax(logits_rot_270, dim=1)
        p_augs.append(p_aug_270)
      assert len(p_augs) == 2

      p_mixture = torch.clamp((p_clean + p_augs[0] + p_augs[1]) / 3.0, 1e-12, 1).log()
      loss_js = ((F.kl_div(p_mixture, p_clean)) + (F.kl_div(p_mixture, p_augs[0])) + (F.kl_div(p_mixture, p_augs[1]))) / 3.0
      self.log_dict({'loss_js': loss_js, 'step': epoch}, on_epoch=True, on_step=False, batch_size=batch_size)
      loss += 10.0 * loss_js

    self.log_dict({"loss": loss, 'step': epoch}, on_epoch=True, on_step=False, batch_size=batch_size)

    return {'loss': loss}

  def training_epoch_end(self, training_step_outputs: List):
    epoch = float(self.trainer.current_epoch) # float required by logger
    # compute average loss over single epoch
    loss_avg = torch.Tensor([x['loss'] for x in training_step_outputs]).mean()
    
    self.logger.experiment.add_scalars("losses", {'train': loss_avg}, epoch)
    self.log_dict({"train_loss": loss_avg, 'step': epoch}, on_epoch=True, on_step=False, sync_dist=False)

    mIoU = self.metric_train_iou.compute()
    self.logger.experiment.add_scalars("mIoU", {'train': mIoU}, epoch)
    self.metric_train_iou.reset()

  def validation_step(self, batch: dict, batch_idx: int):
    if self.xym.device != self.device:
      self.xym = self.xym.to(self.device)

    logits, offsets = self.forward(batch['image'])

    img_height, img_width = batch['image'].shape[2:]
    embeddings = self.xym[:,:,:img_height, :img_width].contiguous() + offsets
   
    loss = self.criterion_semantics(logits, batch['semantics'], mask_keep = batch['semantics'] != 255)

    semantic_preds = torch.argmax(logits, dim=1) # [B x H x W]
    fg = semantic_preds > 0

    self.metric_val_iou(semantic_preds, batch['semantics'])

    # return {'loss': loss}
    return {'loss': loss, 'logits': logits, 'embeddings': embeddings, 'foreground': fg, 'offsets': offsets, 'cmap': self.xym}

  def validation_epoch_end(self, validation_step_outputs: List):
    # compute average loss over single epoch
    loss_avg = torch.Tensor([x['loss'] for x in  validation_step_outputs]).mean() 
    self.logger.experiment.add_scalars("losses", {'val': loss_avg}, self.trainer.current_epoch)

    mIoU = self.metric_val_iou.compute()
    self.logger.experiment.add_scalars("mIoU", {'val': mIoU}, self.trainer.current_epoch)
    self.metric_val_iou.reset()

  def test_step(self, batch: dict, batch_idx: int):
    if self.xym.device != self.device:
      self.xym = self.xym.to(self.device)

    logits, offsets = self.forward(batch['image'])

    # --------- process semantic predictions ---------
    semantic_preds = torch.argmax(logits, dim=1) # [B x H x W]
    self.metric_test_iou(semantic_preds, batch['semantics'])
    fg = semantic_preds > 0
    # fg = batch['semantics'] > 0
    
    # --------- process embeddings ---------
    img_height, img_width = batch['image'].shape[2:]
    
    embeddings = self.xym[:,:,:img_height, :img_width].contiguous() + offsets # [B x 2 x H x W]

    instance_maps = self.cluster_model.cluster(embeddings, fg)
    #self.metric_pq.update(semantic_preds, batch['semantics'], instance_maps, batch['instances'])

    return {'logits': logits, 'embeddings': embeddings, 'foreground': fg, 'offsets': offsets, 'cmap': self.xym, 'instance_maps': instance_maps, 'semantics': semantic_preds}

  def test_epoch_end(self, test_step_outputs: List):
    # --------- compute final metric for semantic segmentation ---------
    mIoU = self.metric_test_iou.compute()
    self.logger.experiment.add_scalars("mIoU", {'test': mIoU}, self.trainer.current_epoch)
    self.metric_test_iou.reset() 

    # --------- compute final metric for panoptic quality ---------
    #pq = self.metric_pq.compute()
    #print("Final panoptic quality: ", pq)
    #print("Final panoptic quality class-wise: ", self.metric_pq.results_classwise)

  def predict_step(self, batch, batch_idx):
    if self.xym.device != self.device:
      self.xym = self.xym.to(self.device)

    logits, offsets = self.forward(batch['image'])
    
    # --------- process semantic predictions ---------
    semantic_preds = torch.argmax(logits, dim=1) # [B x H x W]
    fg = semantic_preds > 0

    # --------- process embeddings ---------
    img_height, img_width = batch['image'].shape[2:]
    embeddings = self.xym[:,:,:img_height, :img_width].contiguous() + offsets # [B x 2 x H x W]

    instance_maps = self.cluster_model.cluster(embeddings, fg)

    return {'logits': logits, 'embeddings': embeddings, 'foreground': fg, 'offsets': offsets, 'cmap': self.xym, 'instance_maps': instance_maps, 'semantics': semantic_preds}

  def lr_scaling(self, current_epoch: int) -> float:
    warm_up_epochs = self.hparams.warm_up_epochs
    
    if current_epoch <= warm_up_epochs:
      lr_scale = current_epoch / warm_up_epochs
    else:
      lr_scale = pow((1 - ((current_epoch - (warm_up_epochs + 1)) / (self.trainer.max_epochs - (warm_up_epochs + 1)))), 2.0)

    return lr_scale

  def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.LambdaLR]]:
    learning_rate = self.hparams.lr
    weight_decay = self.hparams.w_decay

    optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
 
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_scaling)
 
    return [optimizer], [scheduler]