import numpy as np
import torch

def affine_sampler(image, affinecfg):
    '''
    given an image, sample an affine transformation
    '''
    batch_size = image.size(0)
    dims = len(image.shape) - 2
    device = image.device
    if dims == 2:
        theta = affine_sampler_2d(batch_size, device, **affinecfg)
    elif dims == 3:
        theta = affine_sampler_3d(batch_size, device, **affinecfg)
    else:
        raise NotImplementedError
    return theta

def affine_sampler_2d(batch_size: int, 
                        device: str = 'cuda', 
                        rot_range=90,
                        scale_range = [0.9, 1.1],
                        shear_range = 15,
                        translation_range = 0.1,
                        **kwargs
                    ) -> torch.Tensor:
    """Generate random affine transformation parameters."""        
    # Initialize identity matrices for the batch
    theta = torch.eye(2, 3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Random rotation
    angle = torch.empty(batch_size, device=device).uniform_(-rot_range, rot_range)
    angle_rad = angle * np.pi / 180
    cos = torch.cos(angle_rad)
    sin = torch.sin(angle_rad)
    
    rotation_matrix = torch.zeros(batch_size, 2, 2, device=device)
    rotation_matrix[:, 0, 0] = cos
    rotation_matrix[:, 0, 1] = -sin
    rotation_matrix[:, 1, 0] = sin
    rotation_matrix[:, 1, 1] = cos
    
    # Random scale
    scale = torch.empty(batch_size, device=device).uniform_(*scale_range)
    scale_matrix = torch.eye(2, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    scale_matrix = scale_matrix * scale.view(-1, 1, 1)
    
    # Random shear
    shear = torch.empty(batch_size, device=device).uniform_(-shear_range, shear_range)
    shear_rad = shear * np.pi / 180
    shear_matrix = torch.eye(2, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    shear_matrix[:, 0, 1] = torch.tan(shear_rad)
    
    # Random translation
    translation = torch.empty(batch_size, 2, device=device).uniform_(-translation_range, translation_range)
    
    # Combine transformations
    theta[:, :2, :2] = torch.bmm(torch.bmm(rotation_matrix, scale_matrix), shear_matrix)
    theta[:, :2, 2] = translation
    
    return theta

def quart_to_rot(q):
    """
    Convert a quaternion to a rotation matrix.
    
    Args:
        q (torch.Tensor): Quaternion tensor of shape (batch_size, 4).
        
    Returns:
        torch.Tensor: Rotation matrix of shape (batch_size, 3, 3).
    """
    batch_size = q.shape[0]
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    rot_matrix = torch.zeros((batch_size, 3, 3), device=q.device)
    rot_matrix[:, 0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
    rot_matrix[:, 0, 1] = 2 * (qx * qy - qz * qw)
    rot_matrix[:, 0, 2] = 2 * (qx * qz + qy * qw)

    rot_matrix[:, 1, 0] = 2 * (qx * qy + qz * qw)
    rot_matrix[:, 1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
    rot_matrix[:, 1, 2] = 2 * (qy * qz - qx * qw)

    rot_matrix[:, 2, 0] = 2 * (qx * qz - qy * qw)
    rot_matrix[:, 2, 1] = 2 * (qy * qz + qx * qw)
    rot_matrix[:, 2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)
    return rot_matrix

# TODO: Make sure this is correct
def affine_sampler_3d(batch_size: int, 
                      device: str = 'cuda', 
                      rot_range=90,
                      scale_range = [0.9, 1.1],
                      shear_range = 15,
                      translation_range = 0.1,
                      **kwargs
                    ) -> torch.Tensor:
    """Generate random affine transformation parameters in 3D."""
    
    # Initialize identity matrices for the batch
    theta = torch.eye(3, 4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Random rotations around each axis
    angles = torch.empty(batch_size, 1, device=device).uniform_(-rot_range, rot_range)
    angles_rad = angles * np.pi / 180
    # get quaternion
    w = torch.randn(batch_size, 3, device=device)
    w = w / w.norm(p=2, dim=1, keepdim=True)
    sint = torch.sin(angles_rad / 2)
    cost = torch.cos(angles_rad / 2)
    q = torch.cat([cost * torch.ones(batch_size, 1, device=device), w * sint], dim=1)

    rotation_matrix = quart_to_rot(q)
    
    # Random scaling
    scale = torch.empty(batch_size, 3, device=device).uniform_(*scale_range)
    scale_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    scale_matrix = scale_matrix * scale.view(-1, 3, 1)
    
    # Random shearing
    shear = torch.empty(batch_size, 3, device=device).uniform_(-shear_range, shear_range)
    shear_rad = shear * np.pi / 180
    shear_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    shear_matrix[:, 0, 1] = torch.tan(shear_rad[:, 0])
    shear_matrix[:, 0, 2] = torch.tan(shear_rad[:, 1])
    shear_matrix[:, 1, 2] = torch.tan(shear_rad[:, 2])
    
    # Random translation
    translation = torch.empty(batch_size, 3, device=device).uniform_(-translation_range, translation_range)
    
    # Combine transformations
    theta[:, :3, :3] = torch.bmm(torch.bmm(rotation_matrix, scale_matrix), shear_matrix)
    theta[:, :3, 3] = translation
    
    return theta
