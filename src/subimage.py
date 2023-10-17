import numpy as np
from numpy import random
import scipy as scp
import pandas as pd
import torch

def make_subimages(x, window, stride):
    # x shape: (N,C,H,W)
    # n_h: Number of steps in H
    # n_w: Number of steps in W
    N, C, H, W = x.shape
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
    
    for i in range(0,n_h):
        for j in range(0,n_w):
            coordinates.append((i,j))
            subimage = x[:,:,i*stride[0]:i*stride[0]+window[0],j*stride[1]:j*stride[1]+window[1]]
            subimages.append(subimage)
    
    return coordinates, subimages

def make_sample_array(compressed_form, sample_locs):
    N, C, allowed_h, allowed_w = compressed_form.shape    
    dummy_array = np.zeros((N, C, len(sample_locs)))
    for j in range(0,len(sample_locs)):
        dummy_array[:,:,j] = compressed_form[:,:,sample_locs[j][0],sample_locs[j][1]]

    return dummy_array

def make_sample_array_torch(compressed_form, sample_locs):
    N, C, allowed_h, allowed_w = compressed_form.shape    
    dummy_array = torch.zeros((N, C, len(sample_locs)))
    for j in range(0,len(sample_locs)):
        dummy_array[:,:,j] = compressed_form[:,:,sample_locs[j][0],sample_locs[j][1]]

    return dummy_array

def make_coordinates(compressed_form, locations=[(0,0)]):
    sample_list = []
    sample_loc_list = []
    N, C, allowed_h, allowed_w = compressed_form.shape            
    # Generate all samples (one step at a time in H and W direction).
    # Find minimum and maximum H/W locations. We can only move up/down to the limits
    max_h = max([c[0] for c in locations])
    max_w = max([c[1] for c in locations])   
    min_h = min([c[0] for c in locations])
    min_w = min([c[1] for c in locations])        

    if (max_h >= allowed_h) or (max_w >= allowed_w) or (min_h < 0) or (min_w < 0):
        raise Exception('Selected locations are invalid.')            

    # You can either move up/left or down/right based on the number of moves available.
    range_h = np.arange(-min_h, allowed_h - max_h)
    range_w = np.arange(-min_w, allowed_w - max_w)       

    for i in range_h:
        for j in range_w:
            sample_locs = [(l[0]+i,l[1]+j) for l in locations]
            sample_loc_list.append(sample_locs)            
    
    return sample_loc_list

def generate_all_subimage(compressed_form, locations=[(0,0)]):
    sample_list = []
    sample_loc_list = []
    N, C, allowed_h, allowed_w = compressed_form.shape            
    # Generate all samples (one step at a time in H and W direction).
    # Find minimum and maximum H/W locations. We can only move up/down to the limits
    max_h = max([c[0] for c in locations])
    max_w = max([c[1] for c in locations])   
    min_h = min([c[0] for c in locations])
    min_w = min([c[1] for c in locations])        

    if (max_h >= allowed_h) or (max_w >= allowed_w) or (min_h < 0) or (min_w < 0):
        raise Exception('Selected locations are invalid.')            

    # You can either move up/left or down/right based on the number of moves available.
    range_h = np.arange(-min_h, allowed_h - max_h)
    range_w = np.arange(-min_w, allowed_w - max_w)       

    for i in range_h:
        for j in range_w:
            sample_locs = [(l[0]+i,l[1]+j) for l in locations]
            dummy_array = make_sample_array(compressed_form, sample_locs)
            sample_list.append(dummy_array)
            sample_loc_list.append(sample_locs)            
    
    return sample_list, sample_loc_list

def sample_subimage(compressed_form, locations=[(0,0)], n_sample=25):
    sample_list = []
    sample_loc_list = []
    N, C, allowed_h, allowed_w = compressed_form.shape    
        
    max_h = max([c[0] for c in locations])
    max_w = max([c[1] for c in locations])   
    min_h = min([c[0] for c in locations])
    min_w = min([c[1] for c in locations])        

    if (max_h >= allowed_h) or (max_w >= allowed_w) or (min_h < 0) or (min_w < 0):
        raise Exception('Selected locations are invalid.')

    # You can either move up/left or down/right based on the number of moves available.
    range_h = np.arange(-min_h, allowed_h - max_h)
    range_w = np.arange(-min_w, allowed_w - max_w)

    h_N = random.choice(range_h,n_sample-1)
    w_N = random.choice(range_w,n_sample-1)

    # Always include the SAME location in the sample.
    dummy_array = make_sample_array(compressed_form, locations)
    sample_list.append(dummy_array)
    sample_loc_list.append(locations)            

    for i in range(0,n_sample-1):
        sample_locs = [(l[0]+h_N[i],l[1]+w_N[i]) for l in locations]
        dummy_array = make_sample_array(compressed_form, sample_locs)
        sample_list.append(dummy_array)
        sample_loc_list.append(sample_locs)
            
    return sample_list, sample_loc_list

def sample_subimage_torch(compressed_form, locations=[(0,0)], n_sample=25):
    sample_list = []
    sample_loc_list = []
    N, C, allowed_h, allowed_w = compressed_form.shape    
        
    max_h = max([c[0] for c in locations])
    max_w = max([c[1] for c in locations])   
    min_h = min([c[0] for c in locations])
    min_w = min([c[1] for c in locations])        

    if (max_h >= allowed_h) or (max_w >= allowed_w) or (min_h < 0) or (min_w < 0):
        raise Exception('Selected locations are invalid.')

    # You can either move up/left or down/right based on the number of moves available.
    range_h = np.arange(-min_h, allowed_h - max_h)
    range_w = np.arange(-min_w, allowed_w - max_w)

    h_N = random.choice(range_h,n_sample-1)
    w_N = random.choice(range_w,n_sample-1)

    # Always include the SAME location in the sample.
    dummy_array = make_sample_array_torch(compressed_form, locations)
    sample_list.append(dummy_array)
    sample_loc_list.append(locations)            

    for i in range(0,n_sample-1):
        sample_locs = [(l[0]+h_N[i],l[1]+w_N[i]) for l in locations]
        dummy_array = make_sample_array_torch(compressed_form, sample_locs)
        sample_list.append(dummy_array)
        sample_loc_list.append(sample_locs)
            
    return sample_list, sample_loc_list
