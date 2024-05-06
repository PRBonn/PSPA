from abc import ABC, abstractmethod
import torch
from typing import Tuple, Dict
from torchvision import transforms

def image_rgb_split(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """ Extract each channel of a rgb image seperately.

  Args:
      image (torch.Tensor): RGB image of shape [3 x H x W]

  Returns:
      Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Each channel of shape [H x W]
  """
  assert image.ndim == 3

  red = image[0]
  green = image[1]
  blue = image[2]

  return red, green, blue


def channel_stats(channel: torch.Tensor) -> Tuple[float, float]:
  """ Compute mean and std of an image channel 

  Args:
      channel (torch.Tensor): image channel of shape [H x W]

  Returns:
      Tuple[float, float]: mean and std
  """
  assert channel.ndim == 2

  mean = float(channel.mean())
  std = float(channel.std())

  return mean, std


def image_rgb_stack(red: torch.Tensor, green: torch.Tensor, blue: torch.Tensor) -> torch.Tensor:
  assert red.ndim == 2
  assert green.ndim == 2
  assert blue.ndim == 2

  rgb_image = torch.stack([red, green, blue], dim=0)  # [3 x H x W]

  return rgb_image


class ImageNormalizer(ABC):
  """ Generic representation to normalize input images. 
  """

  @abstractmethod
  def __call__(self, img: torch.Tensor) -> torch.Tensor:
    """ Perform the normalization

    Args:
        img (torch.Tensor): Input image of shape [3 x H x W]

    Returns:
        torch.Tensor: Normalized tensor of shape [3 x H x W]
    """

class SingleImageNormalizer(ImageNormalizer):
  """ Normalize each channel of an image seperateley.
  """

  def __call__(self, img: torch.Tensor) -> torch.Tensor:
    assert torch.all(img >= 0.0)
    assert torch.all(img <= 1.0)

    red, green, blue = image_rgb_split(img)

    r_mean, r_std = channel_stats(red)
    g_mean, g_std = channel_stats(green)
    b_mean, b_std = channel_stats(blue)

    red_normalized = self.normalize_channel(red, r_mean, r_std)
    green_normalized = self.normalize_channel(green, g_mean, g_std)
    blue_normalized = self.normalize_channel(blue, b_mean, b_std)

    img_normalized = image_rgb_stack(red_normalized, green_normalized, blue_normalized)

    return img_normalized

  @staticmethod
  def normalize_channel(channel: torch.Tensor, mean: float, std: float):
    """ Compute z-score of single channel """

    channel_normalized = (channel - mean) / (std + 1e-12)

    return channel_normalized
  
class PlusMinusOne(ImageNormalizer):
  """ Normalize image to in [-1, 1].
  """

  def __call__(self, img: torch.Tensor) -> torch.Tensor:
    assert torch.all(img >= 0.0)
    assert torch.all(img <= 1.0)

    img_normalized = -1 + img * 2

    return img_normalized

class GlobalImageNormalizer(ImageNormalizer):
  """ Normalize over all channels of an image.
  """

  def __call__(self, img: torch.Tensor) -> torch.Tensor:
    assert torch.all(img >= 0.0)
    assert torch.all(img <= 1.0)

    mean = img.mean()
    std = img.std()

    img_normalized = (img - mean) / (std + 1e-12)

    return img_normalized


class KeepImageNormalizer(ImageNormalizer):
  """ Do not apply any normalization.
  """

  def __call__(self, img: torch.Tensor) -> torch.Tensor:
    assert torch.all(img >= 0.0)
    assert torch.all(img <= 1.0)

    return img.clone()

class ImageNetNormalizer(ImageNormalizer):
  def __init__(self):
    self.normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

  def __call__(self, image: torch.Tensor) -> torch.Tensor:
    assert torch.all(image <= 1.0)
    assert torch.all(image >= 0.0)

    return self.normalizer(image)

def get_image_normalizer(cfg: Dict):
  name = cfg['data']['image_normalizer']['name']
  if name == 'single_image_normalizer':
    return SingleImageNormalizer()

  if name == 'global_image_normalizer':
    return GlobalImageNormalizer()

  if name == 'keep_image_normalizer':
    return KeepImageNormalizer()
  
  if name == 'imagenet_image_normalizer':
    return ImageNetNormalizer()
  
  if name == 'plus_minus_one':
    return PlusMinusOne()

  assert False, "The specified image normalizer is not implemented."