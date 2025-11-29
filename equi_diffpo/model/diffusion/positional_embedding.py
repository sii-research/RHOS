import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, max_period=10000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
    # def forward(
    #     self,
    #     pos: torch.Tensor,
    #     min_period: float = 3.0e-3,
    #     max_period: float = 4.0,
    # ) -> torch.Tensor:
    #     """
    #     Computes sine-cosine positional embedding vectors for scalar positions.
    #     This is a PyTorch implementation of the JAX version.

    #     Args:
    #         pos (torch.Tensor): A 1D tensor of positions of shape `(b,)`.
    #         embedding_dim (int): The total dimension of the embedding. Must be even.
    #         min_period (float): The minimum period for the sinusoids.
    #         max_period (float): The maximum period for the sinusoids.

    #     Returns:
    #         torch.Tensor: The positional embeddings of shape `(b, embedding_dim)`.
    #     """

    #     # 2. Create the fractions for the geometric progression of periods
    #     # Shape: (embedding_dim / 2)
    #     half_dim = self.dim // 2
    #     fractions = torch.linspace(
    #         0.0, 1.0, half_dim, device=pos.device, dtype=pos.dtype
    #     )

    #     # 3. Calculate the periods for each frequency
    #     # Shape: (embedding_dim / 2)
    #     periods = min_period * (max_period / min_period) ** fractions

    #     # 4. Calculate the angular frequencies (2π / period)
    #     # Shape: (embedding_dim / 2)
    #     angular_freqs = (2.0 * math.pi) / periods

    #     # 5. Calculate the arguments for the sin/cos functions (pos * freq)
    #     # This is an outer product between positions and frequencies.
    #     # pos shape: (b), angular_freqs shape: (d/2)
    #     # Resulting shape: (b, d/2)
    #     sinusoid_input = pos.unsqueeze(1) * angular_freqs.unsqueeze(0)

    #     # 6. Concatenate the sine and cosine of the inputs
    #     # Shape: (b, embedding_dim)
    #     emb = torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)

    #     return emb
