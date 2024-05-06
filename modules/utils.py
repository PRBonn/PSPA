import torch
import torch.nn.functional as F


# x_filter = [[0, 0, 0], [0, -1, 1], [0, 0, 0]]
# y_filter = [[0, 0, 0], [0, -1, 0], [0, 1, 0]]
class GradientFilter():
  def __init__(self, kernel_size: str = "3x3"):
    if kernel_size == "3x3":
      x_filter = [[-0.1250, 0, 0.1250], [-0.2500, 0, 0.2500], [-0.1250, 0, 0.125]]
      y_filter = [[-0.1250, -0.2500, -0.1250], [0, 0, 0], [0.1250, 0.2500, 0.1250]]
    elif kernel_size == "5x5":
      x_filter = [
                 [-0.0078, -0.0156, 0.0, 0.0156, 0.0078],
                 [-0.0312, -0.0625, 0.0, 0.0625, 0.0312],
                 [-0.0469, -0.0938, 0.0, 0.0938, 0.0469],
                 [-0.0312, -0.0625, 0.0, 0.0625, 0.0312],
                 [-0.0078, -0.0156, 0.0, 0.0156, 0.0078]
                 ]
      y_filter = [
                 [-0.0078, -0.0312, -0.0469, -0.0312, -0.0078],
                 [-0.0156, -0.0625, -0.0938, -0.0625, -0.0156],
                 [0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0156, 0.0625, 0.0938, 0.0625, 0.0156],
                 [0.0078, 0.0312, 0.0469, 0.0312, 0.0078]
                 ]
    elif kernel_size == "7x7":
      x_filter = [
                 [-0.0005, -0.002 , -0.0024,  0.0, 0.0024,  0.002 ,  0.0005],
                 [-0.0029, -0.0117, -0.0146,  0.0, 0.0146,  0.0117,  0.0029],
                 [-0.0073, -0.0293, -0.0366,  0.0, 0.0366,  0.0293,  0.0073],
                 [-0.0098, -0.0391, -0.0488,  0.0, 0.0488,  0.0391,  0.0098],
                 [-0.0073, -0.0293, -0.0366,  0.0, 0.0366,  0.0293,  0.0073],
                 [-0.0029, -0.0117, -0.0146,  0.0, 0.0146,  0.0117,  0.0029],
                 [-0.0005, -0.002 , -0.0024,  0.0, 0.0024,  0.002 ,  0.0005]
                 ]
      y_filter = [
                 [-0.0005, -0.0029, -0.0073, -0.0098, -0.0073, -0.0029, -0.0005],
                 [-0.002 , -0.0117, -0.0293, -0.0391, -0.0293, -0.0117, -0.002 ],
                 [-0.0024, -0.0146, -0.0366, -0.0488, -0.0366, -0.0146, -0.0024],
                 [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
                 [ 0.0024,  0.0146,  0.0366,  0.0488,  0.0366,  0.0146,  0.0024],
                 [ 0.002 ,  0.0117,  0.0293,  0.0391,  0.0293,  0.0117,  0.002 ],
                 [ 0.0005,  0.0029,  0.0073,  0.0098,  0.0073,  0.0029,  0.0005]
                 ]
    else:
      raise ValueError(f"The kernel size {kernel_size} is not implemented")
      
    self.x_kernel = torch.tensor(x_filter, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1 x 1 x n_dim x n_dim]
    self.y_kernel = torch.tensor(y_filter, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1 x 1 x n_dim x n_dim]

  def to_device(self, device: torch.device) -> None:
    if self.x_kernel.device != device:
      self.x_kernel = self.x_kernel.to(device)

    if self.y_kernel.device != device:
      self.y_kernel = self.y_kernel.to(device)

  def partial_x(self, x_input: torch.Tensor, x_step_size: float) -> torch.Tensor:
    """ Computes the partial derivative in x-direction.

    Args:
        x_input (torch.Tensor): Compute derivate for this input ... of shape [1 x 1 x H x W]
        x_step_size (float): Step size in x-direction.

    Returns:
        torch.Tensor: Contains the partial derivates in x-direction .. of shape [1 x 1 x H x W]
    """
    self.to_device(device=x_input.device)

    partial_x = F.conv2d(x_input, self.x_kernel, padding='same')
    partial_x /= x_step_size

    return partial_x

  def partial_y(self, y_input: torch.Tensor, y_step_size: float) -> torch.Tensor:
    """ Computes the partial derivative in y-direction.

    Args:
        y_input (torch.Tensor): Compute derivate for this input ... of shape [1 x 1 x H x W]
        y_step_size (float): Step size in y-direction.

    Returns:
        torch.Tensor: Contains the partial derivates in y-direction ... of shape [1 x 1 x H x W]
    """
    self.to_device(device=y_input.device)

    partial_y = F.conv2d(y_input, self.y_kernel, padding='same')
    partial_y /= y_step_size

    return partial_y

class Discrete2DVectorFieldAnalyst():
  def __init__(self, x_step_size: float, y_step_size: float, gradient_filter: GradientFilter) -> None:
    """ Constructor.

    Args:
        x_step_size (float): Step size of discrete grid in x-direction
        y_step_size (float): Step size of discrete grid in y-direction
        gradient_filter (GradientFilter): Operator that can compute partial derivates on discrete vector fields
    """
    """ 

    Args:
        x_step_size (float): 
        y_step_size (float): 
    """
    self.x_step_size = x_step_size
    self.y_step_size = y_step_size
    self.gradient_filter = gradient_filter

  def compute_dxdx(self, vector_field: torch.Tensor) -> torch.Tensor:
    """ Compute the partial derivate of the x-components w.r.t. x.

    Args:
        vector_field (torch.Tensor): Vector field containing the x- and y-components ... of shape [B x 2 x H x W]

    Returns:
        torch.Tensor: Partial derivate of the x-components w.r.t. x ... of shape [B x 1 x H x W]
    """ 

    # Get x-components only
    x_components = vector_field[:,0,:,:].unsqueeze(1) # [B x 1 x H x W]

    return self.gradient_filter.partial_x(x_components, self.x_step_size)

  def compute_dydy(self, vector_field: torch.Tensor) -> torch.Tensor:
    """ Compute the partial derivate of the y-components w.r.t. y.

    Args:
        vector_field (torch.Tensor): Vector field containing the x- and y-components ... of shape [B x 2 x H x W]

    Returns:
        torch.Tensor: Partial derivate of the y-components w.r.t. y ... of shape [B x 1 x H x W]
    """ 

    # Get y-component only
    y_components = vector_field[:,1,:,:].unsqueeze(1) # [B x 1 x H x W]

    return self.gradient_filter.partial_y(y_components, self.y_step_size)

  def compute_dxdy(self, vector_field: torch.Tensor) -> torch.Tensor:
    """ Compute the partial derivate of the x-components w.r.t. y.

    Args:
        vector_field (torch.Tensor): Vector field containing the x- and y-components ... of shape [B x 2 x H x W]

    Returns:
        torch.Tensor: Partial derivate of the x-components w.r.t. y ... of shape [B x 1 x H x W]
    """ 

    # Get x-component only
    x_components = vector_field[:,0,:,:].unsqueeze(1) # [B x 1 x H x W]

    return self.gradient_filter.partial_y(x_components, self.y_step_size)

  def compute_dydx(self, vector_field: torch.Tensor) -> torch.Tensor:
    """ Compute the partial derivate of the y-components w.r.t. x.

    Args:
        vector_field (torch.Tensor): Vector field containing the x- and y-components ... of shape [B x 2 x H x W]

    Returns:
        torch.Tensor: Partial derivate of the y-components w.r.t. x ... of shape [B x 1 x H x W]
    """ 

    # Get y-component only
    y_components = vector_field[:,1,:,:].unsqueeze(1) # [B x 1 x H x W]

    return self.gradient_filter.partial_x(y_components, self.x_step_size)