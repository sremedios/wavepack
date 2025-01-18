import torch
import numpy as np
import torch.nn.functional as F
import pywt
import math
import itertools


# Precompute wavelet filters once
def get_wavelet_filters(wavelet_name='db1', device='cpu'):
    wavelet = pywt.Wavelet(wavelet_name)
    lo = torch.tensor(wavelet.dec_lo, device=device).flip(dims=(0,)).float().view(1,1,-1)
    hi = torch.tensor(wavelet.dec_hi, device=device).flip(dims=(0,)).float().view(1,1,-1)
    lo_rec = torch.tensor(wavelet.rec_lo, device=device).float().view(1,1,-1)
    hi_rec = torch.tensor(wavelet.rec_hi, device=device).float().view(1,1,-1)
    return lo, hi, lo_rec, hi_rec

# Then in your decomposition code, you do not rebuild them:
def decompose_1d(data, lo, hi):
    # data is shape (N, 1, L) for 1D
    out_lo = F.conv1d(data, lo, stride=2)
    out_hi = F.conv1d(data, hi, stride=2)
    return out_lo, out_hi

def recompose_1d(out_lo, out_hi, lo_rec, hi_rec):
    rec_lo = F.conv_transpose1d(out_lo, lo_rec, stride=2)
    rec_hi = F.conv_transpose1d(out_hi, hi_rec, stride=2)
    return rec_lo + rec_hi

def decompose_along_dim(x, dim, lo, hi):
    # 1) Move `dim` to the last dimension via permutation
    total_dims = x.ndim  
    perm = list(range(total_dims))
    perm.append(perm.pop(dim))  # move `dim` to end

    x_perm = x.permute(*perm)   # shape (..., length)

    # 2) Flatten everything except the first two dims (N, C) so we can
    #    call decompose_1d on shape (N*C*extra, 1, length).
    shape_perm = x_perm.shape
    N, C = shape_perm[0], shape_perm[1]
    L = shape_perm[-1]
    
    # Product of any intermediate dims
    intermediate = shape_perm[2:-1]
    extra_size = 1
    for s in intermediate:
        extra_size *= s

    # Reshape => (N*C*extra_size, 1, L)
    x_reshaped = x_perm.reshape(N * C * extra_size, 1, L)

    # 3) Decompose using the precomputed filters
    lo_coeffs, hi_coeffs = decompose_1d(x_reshaped, lo, hi)
    # lo_coeffs, hi_coeffs: shape (N*C*extra_size, 1, L/2)

    # 4) Reshape back to (N, C, *intermediate, L/2)
    new_length = lo_coeffs.shape[-1]
    lo_perm = lo_coeffs.reshape(N, C, *intermediate, new_length)
    hi_perm = hi_coeffs.reshape(N, C, *intermediate, new_length)

    # 5) Invert the permutation so dimension `dim` returns to its original place
    inv_perm = [0]*total_dims
    for i, p in enumerate(perm):
        inv_perm[p] = i

    lo_final = lo_perm.permute(*inv_perm)
    hi_final = hi_perm.permute(*inv_perm)

    return lo_final, hi_final

def recompose_along_dim(lo, hi, dim, lo_rec, hi_rec):
    """
    Invert a one-level 1D wavelet decomposition (lo, hi) along dimension `dim`,
    matching `decompose_along_dim(x, dim, lo, hi)`.

    lo, hi: Tensors each with the same shape as x had after being halved along `dim`.
    dim:  The dimension along which we want to invert the wavelet transform.
    lo_rec, hi_rec: Precomputed reconstruction filters of shape (1,1,filter_length),
                    e.g. from get_wavelet_filters(...).

    Returns:
      A tensor of the same shape as the original x (before decomposition).
    """

    # 1) Move `dim` to the last dimension (just like we did in decompose_along_dim)
    total_dims = lo.ndim
    perm = list(range(total_dims))
    perm.append(perm.pop(dim))  # move `dim` to the end

    lo_perm = lo.permute(*perm)  # shape (..., length//2)
    hi_perm = hi.permute(*perm)  # shape (..., length//2)

    # For example, shape_perm might be (N, C, ..., length//2)
    shape_perm = lo_perm.shape  
    N, C = shape_perm[0], shape_perm[1]
    length_half = shape_perm[-1]

    # 2) Flatten all other dims so we do a 1D inverse transform:
    #    shape => (N*C*extra_size, 1, length_half)
    intermediate = shape_perm[2:-1]  # dims between C and the last dimension
    extra_size = 1
    for s in intermediate:
        extra_size *= s

    lo_reshaped = lo_perm.reshape(N*C*extra_size, 1, length_half)
    hi_reshaped = hi_perm.reshape(N*C*extra_size, 1, length_half)

    # 3) Inverse 1D wavelet transform with conv_transpose1d
    rec_lo = F.conv_transpose1d(lo_reshaped, lo_rec, stride=2)
    rec_hi = F.conv_transpose1d(hi_reshaped, hi_rec, stride=2)

    # 4) Sum partial reconstructions (lo + hi)
    rec_reshaped = rec_lo + rec_hi  # shape => (N*C*extra_size, 1, length_full)

    # 5) Reshape back to (N, C, *intermediate, length_full)
    length_full = rec_reshaped.shape[-1]
    rec_perm = rec_reshaped.reshape(N, C, *intermediate, length_full)

    # 6) Invert the permutation so dimension `dim` goes back to its original place
    inv_perm = [0]*total_dims
    for i, p in enumerate(perm):
        inv_perm[p] = i
    rec_final = rec_perm.permute(*inv_perm)

    return rec_final

def decompose_nd_slow(x, lo, hi):
    dims = len(x.shape) - 2

    # Start with a single sub-band: the entire tensor.
    current_subbands = [x]

    for d in dims:
        new_subbands = []
        # Decompose each existing sub-band along dimension d
        for sb in current_subbands:
            lo_sb, hi_sb = decompose_along_dim(sb, d, lo, hi)
            new_subbands.append(lo_sb)
            new_subbands.append(hi_sb)
        current_subbands = new_subbands

    # After going through all dims, we have 2^len(dims) sub-bands
    return current_subbands

def recompose_nd_slow(subbands, lo_rec, hi_rec):
    dims = len(subbands[0].shape) - 2

    current_subbands = subbands  # list of Tensors
    # number of subbands = 2^len(dims)

    # We recompose in reverse order of dims
    for d in reversed(dims):
        new_subbands = []
        # We'll group subbands in pairs: (lo_sb, hi_sb)
        for i in range(0, len(current_subbands), 2):
            lo_sb = current_subbands[i]
            hi_sb = current_subbands[i+1]
            # Recompose along dimension d
            new_sb = recompose_along_dim(lo_sb, hi_sb, d, lo_rec, hi_rec)
            new_subbands.append(new_sb)
        current_subbands = new_subbands

    # In the end, we have a single sub-band => the original volume
    return current_subbands[0]
    
def decompose_nd(x, lo, hi):
    C = x.shape[1]
    ndims = len(x.shape) - 2
    k = build_wavelet_kernel_nd(lo.squeeze(), hi.squeeze(), ndims=ndims)
    k = k.repeat(C, *[1 for _ in k.shape[1:]])
    k = k.view(k.shape[0]*C, C, *k.shape[2:])
    
    match ndims:
        case 1: return F.conv1d(x, k, stride=2, groups=C)
        case 2: return F.conv2d(x, k, stride=2, groups=C)
        case 3: return F.conv3d(x, k, stride=2, groups=C)
        case _: return decompose_nd_slow(x, dims=ndims, lo=low, hi=hi)


def recompose_nd(children, lo_rec, hi_rec):
    x = torch.cat(children, dim=1)  # e.g. (N, 4*C, H/2, W/2) if ndims=2

    N, sumC, *spatial = x.shape
    ndims = len(spatial)  # 1,2, or 3 for a single-pass wavelet
    subbands = 2 ** ndims
    C = sumC // subbands  # e.g. 4*C / 4 = C

    k = build_wavelet_kernel_nd(lo_rec.squeeze(), hi_rec.squeeze(), ndims=ndims)
    k = k.repeat(C, *[1 for _ in k.shape[1:]])
    k = k.view(k.shape[0]*C, C, *k.shape[2:])
    
    match ndims:
        case 1: return F.conv_transpose1d(x, k, stride=2, groups=C)
        case 2: return F.conv_transpose2d(x, k, stride=2, groups=C)
        case 3: return F.conv_transpose3d(x, k, stride=2, groups=C)
        case _: return recompose_nd_slow(x, dims=ndims, lo=low, hi=hi, max_level=max_level)


def wavelet_packet_decompose_nd_to_leaves(x, lo=None, hi=None, level=0, max_level=1, leaves=None):
    if leaves is None:
        leaves = []
        
    # Base case: if we've reached max_level, store x as a 'leaf'
    if level >= max_level:
        leaves.append(x)
        return leaves

    # 1) Decompose current x by one level across all dims
    subbands = decompose_nd(x, lo=lo, hi=hi)
    C = x.shape[1]
    subbands = torch.split(subbands, C, dim=1)

    # 2) For a wavelet *packet*, we recursively decompose *every* sub-band
    next_level = level + 1
    for sb in subbands:
        wavelet_packet_decompose_nd_to_leaves(
            sb, lo=lo, hi=hi,
            level=next_level, max_level=max_level, leaves=leaves
        )

    return leaves

def wavepack_dec(x, lo=None, hi=None, max_level=1):
    leaves = wavelet_packet_decompose_nd_to_leaves(x, lo=lo, hi=hi, level=0, max_level=max_level)
    stacked = torch.cat(leaves, dim=1)
    return stacked


def wavelet_packet_recompose_nd_from_leaves(leaves, lo_rec=None, hi_rec=None, level=0, max_level=1):
    if level >= max_level:
        return leaves.pop(0)

    ndims = len(leaves[0].shape) - 2
    subbands = []
    for _ in range(2 ** ndims):
        child = wavelet_packet_recompose_nd_from_leaves(
            leaves, lo_rec=lo_rec, hi_rec=hi_rec, 
            level=level+1, max_level=max_level
        )
        subbands.append(child)
    
    # subbands is a list of Tensors => pass to `recompose_nd`
    x_merged = recompose_nd(subbands, lo_rec, hi_rec)
    return x_merged

def wavepack_rec(x_stacked, lo_rec=None, hi_rec=None, max_level=1, original_channels=1):
    # Number of final leaves for wavelet-packet:
    # In ND, each level => 2^(len(dims)) expansions. Over max_level => 2^(len(dims)*max_level).
    num_leaves = 2 ** ((len(x_stacked.shape) - 2) * max_level)

    # x_stacked is (N, C*num_leaves, ...)
    N = x_stacked.shape[0]

    # Split into final sub-bands
    leaves_list = []
    chunk_size = original_channels
    for i in range(num_leaves):
        c_start = i * chunk_size
        c_end = (i + 1) * chunk_size
        leaf = x_stacked[:, c_start:c_end, ...]
        leaves_list.append(leaf)

    # Now recompose from leaves
    leaves_list_m = leaves_list[:]  # shallow copy so we can pop
    x_rec = wavelet_packet_recompose_nd_from_leaves(
        leaves_list_m, lo_rec=lo_rec, hi_rec=hi_rec,
        level=0, max_level=max_level
    )
    return x_rec


def build_wavelet_kernel_nd(lo, hi, ndims=2):
    combos = list(itertools.product([lo, hi], repeat=ndims))

    kernels = []
    for combo in combos:
        k = combo[0]  # shape (L,)
        for c in combo[1:]:
            k = k.unsqueeze(-1)
            c = c.unsqueeze(0)
            k = k * c
        k = k.unsqueeze(0).unsqueeze(1)  # (1, 1, [L, L, ...])
        kernels.append(k)
    
    return torch.cat(kernels, dim=0)
