import torch
import math


def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    This function implements the time embedding block for the timesteps T.

    Parameters
    ----------
    timesteps : torch.Tensor
        Timesteps for the diffusion model

    embedding_dim : int
        Dimension of the embedding

    Returns
    -------
    torch.Tensor
        Time embedding for the timesteps T
    """

    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2

    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.type(torch.float32)[:, None] * emb[None, :]
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], axis=1)

    if embedding_dim % 2 == 1:
        emb = torch.pad(emb, [[0, 0], [0, 1]])

    assert emb.shape == (timesteps.shape[0], embedding_dim), f"{emb.shape}"
    return emb
