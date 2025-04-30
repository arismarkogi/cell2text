import torch.nn as nn
import torch

class LinearProjectionLayer(nn.Module):
    """
    Creates a linear projection layer using nn.Linear.

    Args:
        input_dim (int): The dimension of the input to the projection layer.
        output_dim (int): The dimension of the output from the projection layer.
        bias (bool, optional): If set to False, the layer will not learn an additive bias.
            Default: True
    """
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the linear projection.

        Args:
            x (torch.Tensor): The input tensor to project.

        Returns:
            torch.Tensor: The projected tensor.
        """
        return self.projection(x)
