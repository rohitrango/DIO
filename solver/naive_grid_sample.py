''' Adapted from GPT4 (and corrected to the best of my knowledge) '''
import torch
from time import time

def bilinear_grid_sampling(input, grid):
    """
    Manually implement grid sampling in PyTorch.

    Args:
    - input (Tensor): Input tensor of shape (N, C, H, W)
    - grid (Tensor): Grid tensor of shape (N, H_out, W_out, 2), containing normalized coordinates in [-1, 1]

    Returns:
    - Tensor: Sampled tensor of shape (N, C, H_out, W_out)
    """
    N, C, H, W = input.shape
    _, H_out, W_out, _ = grid.shape
    # Convert coordinates to absolute positions
    # first coordinate is x, second is y, but indexing is done as image[y, x]
    grid = 0.5 * ((grid + 1.0) * torch.tensor([W - 1, H - 1]).view(1, 1, 1, 2).to(grid.device))
    i, j = grid[..., 1], grid[..., 0]
    # i, j = grid[..., 0], grid[..., 1]

    with torch.no_grad():
        i0 = torch.floor(i).clamp(0, H - 1)
        i1 = (i0 + 1).clamp(1, H - 1)
        j0 = torch.floor(j).clamp(0, W - 1)
        j1 = (j0 + 1).clamp(1, W - 1)
        # Gather pixels in four corners  (each of size (N, H_out, W_out))
        batch_idx = torch.arange(N).view(N, 1, 1).expand(N, H_out, W_out).long()
        i0 = i0.long()
        i1 = i1.long()
        j0 = j0.long()
        j1 = j1.long()
    a = input[batch_idx, :, i0, j0]
    b = input[batch_idx, :, i1, j0]
    c = input[batch_idx, :, i0, j1]
    d = input[batch_idx, :, i1, j1]

    # Calculate interpolation weights
    wa = ((i1 - i) * (j1 - j)).unsqueeze(-1)
    wc = ((i1 - i) * (j - j0)).unsqueeze(-1)
    wb = ((i - i0) * (j1 - j)).unsqueeze(-1)
    wd = ((i - i0) * (j - j0)).unsqueeze(-1)
    # Sum up the weighted pixels
    output = wa * a + wb * b + wc * c + wd * d
    output = output.permute(0, 3, 1, 2)
    return output

if __name__ == "__main__":
    # Example usage
    from torch.nn import functional as F
    N, C, H, W = 1, 3, 128, 128  # Input dimensions
    # H_out, W_out = 10, 10  # Output dimensions
    H_out, W_out = 50, 50
    # Create an example input tensor
    input_tensor = torch.randn(N, C, H, W)
    # Create a grid for sampling
    # grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, H_out), torch.linspace(-1, 1, W_out))
    # grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).repeat(N, 1, 1, 1)  # Repeat grid for N batches
    grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0), [N, C, H_out, W_out], align_corners=False).requires_grad_(True)
    print(grid)
    # Sample using the manual grid sampling function
    output_tensor = bilinear_grid_sampling(input_tensor, grid)
    print(output_tensor.shape)  # Output: torch.Size([1, 3, 10, 10])
    output_tensor2 = F.grid_sample(input_tensor, grid, align_corners=True)
    print(output_tensor2.shape)
    # print()
    print(torch.abs(output_tensor - output_tensor2).mean()/torch.abs(output_tensor2).mean())
    # backwards
    t1 = time()
    b1 = torch.autograd.grad(output_tensor.mean(), grid)
    t2 = time()
    b2 = torch.autograd.grad(output_tensor2.mean(), grid)
    t3 = time()
    print(torch.abs(b1[0] - b2[0]).mean()/torch.abs(b2[0]).mean())
    print(t2-t1, t3-t2)