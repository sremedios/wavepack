import torch
import numpy as np
import torch.nn.functional as F
import pywt
import math
import itertools


# Precompute wavelet filters once
def get_wavelet_filters(wavelet_name="db1", device="cpu"):
    wavelet = pywt.Wavelet(wavelet_name)
    lo = torch.tensor(wavelet.dec_lo, device=device).flip(dims=(0,)).float().view(1, 1, -1)
    hi = torch.tensor(wavelet.dec_hi, device=device).flip(dims=(0,)).float().view(1, 1, -1)

    lo_rec = torch.tensor(wavelet.rec_lo, device=device).float().view(1, 1, -1)
    hi_rec = torch.tensor(wavelet.rec_hi, device=device).float().view(1, 1, -1)
    return lo, hi, lo_rec, hi_rec


def zero_insert_1d(k, stride=2):
    """
    Zero-insert (upsample) a 1D kernel by 'stride'.
    
    If k is shape (L,) or any shape that flattens to (L,),
    the result is shape (1,1,L_up).
    
    L_up = (L - 1)*stride + 1
    
    Example:
      k = [1, 2, 3], stride=2
      => [1, 0, 2, 0, 3]  => length 5
      => final shape (1,1,5)
    """
    # Flatten to shape (L,)
    k = k.flatten()      # if k was (1,1,L) or (L,), now shape(L,)
    L = k.shape[0]
    
    # Compute the length after zero-inserting
    L_up = (L - 1)*stride + 1
    
    # Create an all-zero array
    k_up = k.new_zeros((L_up,))  # shape (L_up,)
    
    # Place original samples at intervals of 'stride'
    k_up[0::stride] = k
    
    # Return shape => (1,1,L_up)
    return k_up.unsqueeze(0).unsqueeze(0)

def cascade_1d_kernels(kernels):
    # Successive zero insertion
    ks_up = [zero_insert_1d(k, stride=2**i) for i, k in enumerate(kernels)]

    # Cascade all kernels together
    out = ks_up[0]
    for k_next in ks_up[1:]:
        ksize_next = k_next.shape[-1]
        out = F.conv1d(out, k_next, padding=ksize_next - 1)
    
    return out

def multi_level_1d_paths(lo, hi, levels):
    """
    Return a list of cascaded 1D kernels for all wavelet paths
    in 'levels' 1D decomposition.
    E.g. for levels=2 => paths = [lo->lo, lo->hi, hi->lo, hi->hi].
    Returns a list of shape (1,1,L_cascaded).
    """
    return torch.stack([cascade_1d_kernels(p) for p in itertools.product([lo, hi], repeat=levels)])

def outer_product_nd(filters):
    # Start with the flattened first filter
    k = filters[0].flatten()   # shape (L_1,)

    # Multiply stepwise for each subsequent filter
    for f in filters[1:]:
        f_flat = f.flatten()   # shape (L_i,)
        # broadcast outer product
        # k: shape (...), f_flat: shape (L_i,)
        # => shape (..., L_i)
        k = k.unsqueeze(-1) * f_flat.unsqueeze(0)

    # Now k is shape (L_1, L_2, ..., L_n)
    # Put it into (1,1,...) format for a standard ND kernel
    k = k.unsqueeze(0).unsqueeze(1)
    return k

def build_nd_wavepack_kernel(lo, hi, levels, ndims):
    filters = [multi_level_1d_paths(lo, hi, levels) for _ in range(ndims)]
    combos = itertools.product(*filters)
    return torch.cat([outer_product_nd(kernel_list) for kernel_list in combos], dim=0)


def fwpt(x, lo, hi, levels=1):
    """Fast wavelet packet transform"""
    ndims = len(x.shape) - 2
    k = build_nd_wavepack_kernel(lo, hi, levels=levels, ndims=ndims)
    match ndims:
        case 1: return F.conv1d(x, k, stride=2**levels)
        case 2: return F.conv2d(x, k, stride=2**levels)
        case 3: return F.conv3d(x, k, stride=2**levels)
        case _: raise NotImplementedError("Convolutions of higher dimensions are not supported.")


def ifwpt(x, lo, hi, levels=1):
    """Fast wavelet packet transform"""
    ndims = len(x.shape) - 2
    k = build_nd_wavepack_kernel(lo, hi, levels=levels, ndims=ndims)
    match ndims:
        case 1: return F.conv_transpose1d(x, k, stride=2**levels)
        case 2: return F.conv_transpose2d(x, k, stride=2**levels)
        case 3: return F.conv_transpose3d(x, k, stride=2**levels)
        case _: raise NotImplementedError("Convolutions of higher dimensions are not supported.")

