import torch as th
from torch import nn
import random
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import _get_inverse_affine_matrix


def get_params(degrees=(-30, 30), translate=(-0.2, 0.2), scale_ranges=(0.8, 1.2), shears=(-30, 30)):
    """Get parameters for affine transformation
    Returns:
        sequence: params to be passed to the affine transformation
    """
    angle = random.uniform(degrees[0], degrees[1])
    if translate is not None:
        max_dx = translate[0]
        max_dy = translate[1]
        translations = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
    else:
        translations = (0, 0)

    if scale_ranges is not None:
        scale = random.uniform(scale_ranges[0], scale_ranges[1])
    else:
        scale = 1.0

    if shears is not None:
        shear = (random.uniform(shears[0], shears[1]), random.uniform(shears[0], shears[1]))
    else:
        shear = (0.0, 0.0)

    ret = [angle, translations, scale, shear]
    center = (0.5, 0.5)
    aff_mat = _get_inverse_affine_matrix(center, *ret)

    return aff_mat


def get_params_from_coords(batch_size, pert_rate=0.2, dtype=th.float, device="cuda"):
    """Get parameters for affine transformation
    Returns:
        sequence: params to be passed to the affine transformation
    """
    new_upperleft = th.tensor([[[-1.], [-1.]]] * batch_size, dtype=dtype, device=device)
    new_upperright = th.tensor([[[-1.], [1.]]] * batch_size, dtype=dtype, device=device)
    new_downleft = th.tensor([[[1.], [-1.]]] * batch_size, dtype=dtype, device=device)

    upperleft = new_upperleft * (1 - 2 * th.rand_like(new_upperleft) * pert_rate)
    upperright = new_upperright * (1 - 2 * th.rand_like(new_upperright) * pert_rate)
    downleft = new_downleft * (1 - 2 * th.rand_like(new_downleft) * pert_rate)

    coords = th.cat([upperleft, upperright, downleft], dim=2)
    coords = th.cat([coords, th.ones_like(coords)[:, :1]], dim=1)

    new_coords = th.cat([new_upperleft, new_upperright, new_downleft], dim=2)
    new_coords = th.cat([new_coords, th.ones_like(new_coords)[:, :1]], dim=1)

    aff_mat = th.bmm(new_coords, th.inverse(coords))[:, :2]

    return aff_mat


def affine_transform(image, aff_mat):
    grid = F.affine_grid(aff_mat, image.size())
    image_t = F.grid_sample(image, grid)

    return image_t


def compute_cosine_distance(x, y):
    # mean shifting by channel-wise mean of `y`.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mu
    y_centered = y - y_mu

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # channel-wise vectorization
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

    # consine similarity
    cosine_sim = th.bmm(x_normalized.transpose(1, 2),
                           y_normalized)  # (N, H*W, H*W)

    return cosine_sim


def compute_relative_distance(dist_raw):
    dist_min, _ = th.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde


class CSSrcMapper(nn.Module):
    def __init__(self, device="cuda"):
        super(CSSrcMapper, self).__init__()
        feats_dict = th.load('assets/cs_classes_rn50.pt')
        self.classes = feats_dict["classes"]
        self.classes2color = feats_dict["classes2color"]
        self.classes2color = {k: v[None, :, None, None].to(device) for k, v in self.classes2color.items()}
        self.classes2feat = feats_dict["classes2feat"]
        self.classes2feat = {k: v[:, None].to(device) for k, v in self.classes2feat.items()}

    def __call__(self, src):
        src_feat = th.zeros((src.size(0), 1024, src.size(2), src.size(3)), dtype=src.dtype, device=src.device)
        for cls in self.classes:
            idx = ((src * 127.5 + 127.5).long() == self.classes2color[cls]).all(dim=1)
            for i in range(src.size(0)):
                src_feat[i, :, idx[i]] = self.classes2feat[cls].type(src.dtype)
        return src_feat

def leftupper_coords_from_size(size, patch_size=256):
    h, w = size[0], size[1]
    assert (h >= patch_size and w >= patch_size)
    id_h, id_w = (h - 1) // (patch_size // 2), (w - 1) // (patch_size // 2)

    coords_list = []
    for i in range(id_h):
        for j in range(id_w):
            if i < id_h - 1:
                if j < id_w - 1:
                    coords_list.append((i * patch_size // 2 , j * patch_size // 2))
                else:
                    coords_list.append((i * patch_size // 2, w - patch_size // 2))
            else:
                if j < id_w - 1:
                    coords_list.append((h - patch_size // 2, j * patch_size // 2))
                else:
                    coords_list.append((h - patch_size // 2, w - patch_size // 2))

    return coords_list
