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

class MLPProjectionLayer(nn.Module):
    """
    Creates a multi-layer perceptron (MLP) projection layer with GELU activation.

    Args:
        input_dim (int): The dimension of the input to the projection layer.
        hidden_dim (int): The dimension of the hidden layer in the MLP.
        output_dim (int): The dimension of the output from the projection layer.
        dropout_prob (float, optional): The probability of dropout. If 0, dropout is not applied.
            Default: 0.0
        bias (bool, optional): If set to False, the linear layers will not learn an additive bias.
            Default: True
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_prob: float = 0.0, bias: bool = True) -> None:
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        ]
        if dropout_prob > 0:
            layers.append(nn.Dropout(p=dropout_prob))
        layers.extend([
            nn.Linear(hidden_dim, output_dim, bias=bias),
            nn.LayerNorm(output_dim)
        ])
        self.projection = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the MLP projection.

        Args:
            x (torch.Tensor): The input tensor to project.

        Returns:
            torch.Tensor: The projected tensor.
        """
        return self.projection(x)