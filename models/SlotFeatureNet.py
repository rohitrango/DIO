'''
code inspired by SlotAttention type of work

the hypothesis is that even though images are dense (things like labelmaps can be sparse), the underlying 
dynamics are governed by a few key set of points. This is the idea behind SlotFeatureNet (to learn this set of features)
'''
import numpy as np
import torch
from torch.nn import functional as F
from solver.utils import gaussian_1d, img2v_2d, v2img_2d, separable_filtering
from torch.nn.functional import scaled_dot_product_attention
from torch import nn

class SlotFeatureNet(nn.Module):
    def __init__(self, 
                    model,
                    num_queries=100, 
                    num_features=32,  
                    num_att_features=32, 
                    kps_dropout=0,
                    dims=2,
                    lambda_=1e-3,
                    scale_codebookloss = True,
                    use_ln = True,
                    w_mode='attention',
    ):
        '''
        model: the (nn.Module) model to extract features from
        num_queries: number of queries to use for the codebook
        num_features: number of features output from the model
        num_att_features: number of features to use for attention
        kps_dropout: dropout to use to drop keypoints
        dims: image dimensions
        w_mode: mode to compute the weight matrix for polyaffine
            choices: 
                - attention: use attention to compute the weights  w_q(x) = Attention(code(q) * F(x))
                - mahalanobis: learnt mahalanobis distance to compute the weights w_q(x) = exp(-||kps_q - x||_{A_q}^2)
        '''
        super().__init__()
        self.model = model
        self.dims = dims
        self.codebook = nn.Parameter(0.1 * torch.randn(num_queries, num_features))
        self.log_sigma_cb = nn.Parameter(torch.zeros(1))    # parameter to multiply the codebook with
        self.log_scale_v = nn.Parameter(torch.zeros(1)-2)
        self.scale_cbloss = scale_codebookloss
        self.lambda_ = lambda_  

        # normalize the codebook and image features
        self.ln_layer = nn.LayerNorm(num_features, elementwise_affine=False, bias=False) if use_ln else lambda x: x

        # coordinate network attention
        # this is where codebook is the query, and image features and their locations are (key, value) 
        # for each query we get a coordinate
        self.coord_codebook = nn.Linear(num_features, num_att_features, bias=False)
        self.coord_features = nn.Linear(num_features, num_att_features, bias=False)

        # second attention layer, where each feature competes for codebook slots
        self.w_mode = w_mode
        if w_mode == 'attention':
            self.w_codebook = nn.Linear(num_features, num_att_features, bias=False)
            self.w_features = nn.Linear(num_features, num_att_features, bias=False)
        elif w_mode == 'mahalanobis':
            # self.A = nn.Parameter(torch.randn(num_queries, dims, dims))
            self.A = nn.Parameter(torch.eye(dims)[None].expand(num_queries, -1, -1) + 0.01 * torch.randn(num_queries, dims, dims))
            self.logs = nn.Parameter(torch.zeros(1))
        # dropout parameter
        self.kps_dp = kps_dropout

    def forward(self, x):
        # x: [B, C, H, W, D]
        B = x.shape[0]
        x = self.model(x)   # BCHW[D]
        if isinstance(x, list):
            x = x[0]

        # flatten the image features to BSC
        if self.dims == 2:
            flatten_x = x.permute(0, 2, 3, 1).flatten(1, -2)   
        elif self.dims == 3:
            flatten_x = x.permute(0, 2, 3, 4, 1).flatten(1, -2) 

        grid = grid_orig = F.affine_grid(torch.eye(self.dims, self.dims+1, device=x.device)[None].expand(B, -1, -1), x.shape, align_corners=True)
        grid = (grid.flatten(1, -2)) # B, S, dims   

        # get image features for attention 
        kf = self.coord_features(self.ln_layer(flatten_x))
        # codebook attention features
        codebook_att1 = self.coord_codebook(self.ln_layer(self.codebook))[None].expand(B, -1, -1) # [B, Nq, C]
        codebook_coords = scaled_dot_product_attention(codebook_att1[:, None], kf[:, None], grid[:, None], dropout_p=0)[:, 0]   # [B, Nq, 2]

        ret = {
            'flatten_x': flatten_x,             # [B, S, C]
            'size': x.shape,                    # size
            'grid': grid_orig,                  # [B, H, W, D, 3] or [B, H, W, 2]
            'query_coords': codebook_coords     # [B, Nq, 2]
        }
        
        # compute coordinates
        return ret

    def keypoint_spread_loss(self, retdict):
        ''' 
        given the coordinates of the keypoints, compute the spread loss
        '''
        coords = retdict['query_coords']
        B, Nq, dim = coords.shape
        dist = torch.cdist(coords, coords, p=2)  # [B, Nq, Nq]
        return -dist.topk(2, dim=-1, largest=False).values.mean()

    def compute_v(self, fix_retdict, mov_retdict):
        '''
        fix_retdict, mov_retdict: dictionaries returned by the forward pass of the model
        '''
        # retrieve all variables
        fixed_features = fix_retdict['flatten_x']
        fixed_coords = fix_retdict['query_coords']
        moving_coords = mov_retdict['query_coords']
        grid = fix_retdict['grid']
        size = fix_retdict['size']

        # we are transforming fixed coords to moving coords, so t = t_m - t_f
        translations = moving_coords - fixed_coords        # [B, Nq, dim]

        print(moving_coords.min(1).values, moving_coords.max(1).values)
        print(fixed_coords.min(1).values, fixed_coords.max(1).values)

        # batch size
        B = fixed_features.shape[0]
        if self.dims == 3:
            H, W, D = size[2:]
        else:
            H, W = size[2:]

        # B1QC, B1SC
        if self.w_mode == 'attention':
            # get keys and queries
            codebook_keys = self.w_codebook(self.ln_layer(self.codebook))[None].expand(B, -1, -1)
            feature_queries = self.w_features(self.ln_layer(fixed_features))
            # apply attention to get weighted average of translations
            transl_avg = torch.exp(self.log_scale_v) * scaled_dot_product_attention(feature_queries[:, None], codebook_keys[:, None], translations[:, None], dropout_p=0)[:, 0]
        
        elif self.w_mode == 'mahalanobis':
            # in this case, we have to use the grid as features
            flatten_grid = grid.flatten(1, -2) 
            x_minus_q = flatten_grid[:, :, None] - fixed_coords[:, None]   # [B, S, Nq, dim]
            Ax_minus_q = torch.matmul(self.A[None, None], x_minus_q[..., None])[..., 0]  # [B, S, Nq, dim]
            Ax_minus_q = (Ax_minus_q + self.lambda_ * x_minus_q).norm(dim=-1)  # [B, S, Nq]
            sigma = torch.exp(self.logs)
            # compute the weights
            w = F.softmax(-sigma * Ax_minus_q, dim=-1)  # [B, S, Nq]
            transl_avg = (w[..., None] * translations[:, None]).sum(-2) # [B, S, dim]
        
        # convert to grid
        if self.dims == 3:
            transl_avg = transl_avg.reshape(B, H, W, D, -1)
        else:
            transl_avg = transl_avg.reshape(B, H, W, -1)

        return transl_avg
    
    def codebook_orthogonal_loss(self, normalize=True):
        ''' compute the orthogonal loss for the codebook '''
        code = self.codebook # [Nq, C]
        if normalize:
            cct = torch.abs(code @ code.T) / code.norm(dim=-1)[:, None] / code.norm(dim=-1)[None]
        else:
            cct = torch.abs(code @ code.T)
        if self.scale_cbloss:
            cct = cct * torch.exp(self.log_sigma_cb)
        # take log softmax
        loss = F.log_softmax(cct, dim=-1)
        loss = -torch.diag(loss).mean()
        return loss

    def solve_affine(self, fix_coords, mov_coords, keep_p = 0.25):
        """
        Solve for the optimal affine transformation that maps fixed coordinates to moving coordinates.
        Uses least squares to find the transformation matrix A and translation vector b such that:
        mov_coords ≈ fix_coords @ A.T + b
        
        Args:
            fix_coords: Fixed coordinates tensor of shape [B, N, dim]
            mov_coords: Moving coordinates tensor of shape [B, N, dim]
            
        Returns:
            tuple: (A, b) where:
                A: Affine matrix of shape [B, dim, dim]
                b: Translation vector of shape [B, dim]
        """
        # Validate input shapes
        assert fix_coords.shape == mov_coords.shape
        batch_size, num_points, dims = fix_coords.shape
        p = keep_p

        # minimum number of points needed to define an affine transformation (multiplied by 2 for stability)
        minaffpoints = {
            2: 3*2,
            3: 4*2
        }

        if p < 1:
            while True:
                u = np.random.rand(num_points) < p
                if u.sum() > minaffpoints[dims]:
                    break
            keep = np.where(u)[0]
            # crop them out
            fix_coords = fix_coords[:, keep]
            mov_coords = mov_coords[:, keep]
        
        # Center the point sets to handle translation separately
        fix_mean = fix_coords.mean(dim=1, keepdim=True)  # [B, 1, dim]
        mov_mean = mov_coords.mean(dim=1, keepdim=True)  # [B, 1, dim]
        
        fix_centered = fix_coords - fix_mean  # [B, N, dim]
        mov_centered = mov_coords - mov_mean  # [B, N, dim]
        
        # Construct the system of equations for least squares
        # For each batch, solve: fix_centered @ A.T = mov_centered
        
        # Compute components for the least squares solution
        # H = (X^T X)^(-1) X^T Y
        fix_transpose = fix_centered.transpose(1, 2)  # [B, dim, N]
        
        # Compute X^T X and its inverse
        xtx = torch.bmm(fix_transpose, fix_centered)  # [B, dim, dim]
        xtx_inv = torch.inverse(xtx)  # [B, dim, dim]
        
        # Compute X^T Y
        xty = torch.bmm(fix_transpose, mov_centered)  # [B, dim, N] @ [B, N, dim] = [B, dim, dim]
        
        # Solve for the affine matrix A
        A = torch.bmm(xtx_inv, xty).transpose(1, 2)  # [B, dim, dim]
        
        # Solve for the translation vector b
        b = mov_mean.squeeze(1) - torch.bmm(fix_mean, A.transpose(1, 2)).squeeze(1)  # [B, dim]
        A = torch.cat([A, b[..., None]], dim=-1)
        return A 