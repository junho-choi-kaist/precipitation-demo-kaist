import numpy as np
from numpy import random
import scipy as scp
import pandas as pd
import torch

from tqdm import tqdm

from subimage_mask import make_subimage

def l2_distance(original, sample, n_dim=2):
    dim = [i for i in range(1,n_dim)]
    return torch.sum((sample-original)**2,dim=dim)

def l1_distance(original, sample, n_dim=2):
    dim = [i for i in range(1,n_dim)]
    return torch.sum(torch.abs(sample-original),dim=dim)

def cosine_distance(original, sample, n_dim=2):
    dim = [i for i in range(1,n_dim)]
    return (torch.sum(original*sample,dim=dim)/
            torch.sqrt(torch.sum(original**2,dim=dim))/
            torch.sqrt(torch.sum(sample**2,dim=dim)))

def measure_distance(original, sample_list, distance):
    diff_list = []
    n_dim = len(original.shape)
    for sample in sample_list:
        distances = distance(torch.Tensor(original).contiguous(), torch.Tensor(sample).contiguous(), n_dim)
        distances[torch.isnan(distances)] = np.inf        
        diff_list.append(distances)
    
    return diff_list

def find_min_distance(diff_list):
    diff_stack = torch.stack(diff_list,dim=1)
    min_diff, min_ind = torch.min(diff_stack,1)
    
    return min_diff, min_ind

def find_similar_examples(min_diff, topk=5):
    sort_ind = torch.argsort(min_diff,dim=0)
    sort_values = min_diff[sort_ind]
    
    return sort_ind[:topk], sort_values[:topk]

def find_similar_subimages(min_diff, min_ind, sample_loc_list, subimages, coordinates, topk=5):
    sort_ind = torch.argsort(min_diff,dim=0)
    sort_values = min_diff[sort_ind]
    sort_min_ind = min_ind[sort_ind][:topk]
    
    similar_subimage_list = []
    for idx in sort_ind[:topk]:
        min_idx = min_ind[idx]
        locs = sample_loc_list[min_idx]
        min_subimage = []
        for c in locs:
            subimage_idx = coordinates.index(c)
            min_subimage.append(subimages[subimage_idx][idx])
        
        similar_subimage_list.append(min_subimage)    
    return sort_ind[:topk], sort_values[:topk], sort_min_ind[:topk], similar_subimage_list


def find_similar_subimages_dataset(min_diff, min_ind, sample_loc_list, dataset, coordinates, n_h, n_w, topk=5):
    sort_ind = torch.argsort(min_diff,dim=0)
    sort_values = min_diff[sort_ind]
    sort_min_ind = min_ind[sort_ind][:topk]    
    
    similar_subimage_list = []
    for idx in sort_ind[:topk]:
        min_idx = min_ind[idx]
        locs = sample_loc_list[min_idx]

        radar_history, _, mask, _, _ = dataset[idx.item()]
        radar_history = torch.Tensor(np.expand_dims(radar_history,0))
        mask = torch.Tensor(np.expand_dims(mask,0))
        _, target_subimages, _ = make_subimage(radar_history, mask, n_h, n_w, threshold=0.5)            
        
        min_subimage = []
        for c in locs:
            subimage_idx = coordinates.index(c)
            min_subimage.append(target_subimages[subimage_idx][0])        
        similar_subimage_list.append(min_subimage)    
    return sort_ind[:topk], sort_values[:topk], sort_min_ind[:topk], similar_subimage_list


def find_random_subimages(ind, min_ind, sample_loc_list, subimages, coordinates, topk=5):
    sort_ind = ind
    sort_values = ind
    sort_min_ind = min_ind[sort_ind][:topk]
    
    similar_subimage_list = []
    for idx in sort_ind[:topk]:
        min_idx = min_ind[idx]
        locs = sample_loc_list[min_idx]
        min_subimage = []
        for c in locs:
            subimage_idx = coordinates.index(c)
            min_subimage.append(subimages[subimage_idx][idx])
        
        similar_subimage_list.append(min_subimage)    
    return sort_ind[:topk], sort_values[:topk], sort_min_ind[:topk], similar_subimage_list
