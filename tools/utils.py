import torch
import numpy as np

def numpy2torch(var, device):
    var_torch = torch.from_numpy(var).float().to(device)
    var_torch.requires_grad_(requires_grad=False)
    return var_torch