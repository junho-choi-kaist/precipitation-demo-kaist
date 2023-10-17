import numpy as np
from numpy import random
import scipy as scp
import pandas as pd
import torch

def make_subimages(x, window, stride):
    # x shape: (N,C,H,W)
    # n_h: Number of steps in H
    # n_w: Number of steps in W
    # Give upper left & botton right corner coordinates
    # And relevant subimages given the window and stride
    N, C, H, W = x.shape
    
#     ## total number of steps. H/stride_h, W/stride_w
#     n_h = H//stride[0]
#     n_w = W//stride[1]
    
#     # When we multiply back, there may be a difference with original dim
#     r_h = H - stride[0]*n_h
#     r_w = W - stride[1]*n_w
        
    H_leftover = H - window[0]
    W_leftover = W - window[1]
    n_h = H_leftover/stride[0]
    n_w = W_leftover/stride[1]
        
    if (n_h != int(n_h)) or (n_w != int(n_w)):
        raise Exception("Window and stride not applicable to image dimensions.")
    
    n_h = int(n_h)+1
    n_w = int(n_w)+1
    
    coordinates = []
    subimages = []
    
    if (n_h > 0) and (n_w > 0):
        h = 0
        for i in range(0,n_h):
            w = 0
            for j in range(0,n_w):
                coordinates.append([(h,w),(h+window[0],w+window[1])])
                subimage = x[:,:,h:h+window[0],w:w+window[1]]
                subimages.append(subimage)            
                w += stride[1]
            h += stride[0]
    elif (n_h == 0) and (n_w > 0):
        h = 0
        w = 0
        for j in range(0,n_w):
            coordinates.append([(h,w),(h+window[0],w+window[1])])
            subimage = x[:,:,h:h+window[0],w:w+window[1]]
            subimages.append(subimage)            
            w += stride[1]        
    elif (n_h > 0) and (n_w == 0):
        h = 0
        w = 0
        for i in range(0,n_h):
            h += stride[0]            
            coordinates.append([(h,w),(h+window[0],w+window[1])])
            subimage = x[:,:,h:h+window[0],w:w+window[1]]
            subimages.append(subimage)            
    else:
        h = 0
        w = 0
        coordinates.append([(h,w),(h+window[0],w+window[1])])
        subimage = x[:,:,h:h+window[0],w:w+window[1]]
        subimages.append(subimage)            
    
    return coordinates, subimages

def make_sample_array(compressed_form, sample_locs, reshape=False, concat=True):
    b, c, _, _ = compressed_form.shape
    samples = []
    loc = sample_locs[0]
    window = [loc[1][0]-loc[0][0],loc[1][1]-loc[0][1]]
    for loc in sample_locs:        
        if reshape:
            # N x C x window
            samples.append(compressed_form[:,:,loc[0][0]:loc[1][0],loc[0][1]:loc[1][1]].reshape((b,-1)))
        else:
            # N x C x (window product)
            samples.append(compressed_form[:,:,loc[0][0]:loc[1][0],loc[0][1]:loc[1][1]])
    
    if concat:
        return np.concatenate(samples)
    else:
        return samples


def make_sample_array_torch(compressed_form, sample_locs, reshape=False, concat=True):
    b, c, _, _ = compressed_form.shape
    samples = []
    loc = sample_locs[0]
    window = [loc[1][0]-loc[0][0],loc[1][1]-loc[0][1]]
    for loc in sample_locs:
        
        if reshape:
            # N x C x window
            samples.append(compressed_form[:,:,loc[0][0]:loc[1][0],loc[0][1]:loc[1][1]].reshape((b,-1)))
        else:
            # N x C x (window product)
            samples.append(compressed_form[:,:,loc[0][0]:loc[1][0],loc[0][1]:loc[1][1]])
        
    if concat:
        return torch.cat(samples)
    else:
        return samples
                           

