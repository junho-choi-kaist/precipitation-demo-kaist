from itertools import chain
import torch
import os
import gzip
import numpy as np
from datetime import datetime, timedelta
from pyproj import Proj
import glob
from numpy.random import choice

from tqdm import tqdm

import torch.nn.functional as F

from subimage import *
from subimage_square import make_sample_array as make_sample_array_sq


class SlicHiddenStateDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, hidden_state_path, years, 
                 dataset_type = 'train', 
                 window_size = 'small', 
                 target_lead=1,
                ):

        self.data_path = hidden_state_path
        self.dataset_type = dataset_type
        self.window_size = window_size
        self.files = os.listdir(self.data_path+dataset_type+'/'+window_size)
        self.years = [str(y) for y in years]
        self.used_files = [f for f in self.files if f.split('.')[0].split('_')[1] == str(target_lead)]
        self.used_files = [f for f in self.used_files if f.split('.')[0].split('_')[0][:4] in self.years]        
        self.used_files.sort()
        self.data_list = []
        self.dates = [f.split('.')[0].split('_')[0] for f in self.used_files]
              
    def __len__(self) -> int:
        return len(self.used_files)
            
    def __getitem__(self, raw_idx):        
        # hsr & cumul data stack
        fname = self.used_files[raw_idx]
        file_loc = self.data_path+'/'+self.dataset_type+'/'+self.window_size+'/'
        hidden_state = np.load(file_loc+fname,allow_pickle=True)['hidden']

        return hidden_state, raw_idx

class SlicHiddenStateClusterDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, hidden_state_path, file_list, index_path, dataset_type='train', window_size='small',cluster_lvl=[1], cluster_label=[0]):

        self.data_path = hidden_state_path
        self.dataset_type = dataset_type
        self.window_size = window_size
        self.used_files = file_list
        self.dataset_type = dataset_type
        self.index_list = index_list
        conditions = [self.index_list[:,1+cluster_lvl[i]]==cluster_label[i] for i in range(0,len(cluster_lvl))]
        conditions = np.stack(conditions,-1)
        self.used_index_list = self.index_list[np.all(conditions,1),:]
                            
    def __len__(self) -> int:
        return len(self.used_index_list)
            
    def __getitem__(self, raw_idx):        
        # hsr & cumul data stack
        target = self.used_index_list[raw_idx]
        fname = self.used_files[target[0]]
        target_path = self.data_path+'/'+self.dataset_type+'/'+self.window_size+'/'
        hidden_state = np.load(target_path+fname,allow_pickle=True)['hidden']
        
        return hidden_state, raw_idx    


class SlicHiddenStateClusterDatasetV2(torch.utils.data.Dataset):
    """
    """
    def __init__(self, hidden_state_path, file_list, index_path, dataset_type='train', yr_list=None, window_size='small',cluster_lvl=[1], cluster_label=[0]):

        self.data_path = hidden_state_path
        self.index_path = index_path
        self.dataset_type = dataset_type
        self.window_size = window_size
        self.used_files = file_list
        self.dataset_type = dataset_type
        
        self.target_path = self.data_path+'/'+self.dataset_type+'/'+self.window_size+'/'
        
        index_file = np.load(index_path,allow_pickle=True)
        self.index_list = index_file['index']
        self.index_date = index_file['date']
        self.index_bb = index_file['bounding_box']
        
        conditions = [self.index_list[:,1+cluster_lvl[i]]==cluster_label[i] for i in range(0,len(cluster_lvl))]

        # Add condition for yr_list:
        if yr_list is not None:
            yr_conditions = np.array([(d[:4] in yr_list) for d in self.index_date])
            conditions.append(yr_conditions)
                
        conditions = np.stack(conditions,-1)
        self.used_index_list = self.index_list[np.all(conditions,1),:]
                            
    def __len__(self) -> int:
        return len(self.used_index_list)
            
    def __getitem__(self, raw_idx):        
        # hsr & cumul data stack
        target = self.used_index_list[raw_idx]
        fname = self.used_files[target[0]]
        hidden_state = np.load(self.target_path+fname,allow_pickle=True)['hidden']
        
        return hidden_state, raw_idx    

    
    
    
class Unet3HiddenStateDatasetIdx(torch.utils.data.Dataset):
    """
    """
    def __init__(self, hidden_state_path, years, dataset_type:str = 'train', target_lead=1,
                ):

        self.data_path = hidden_state_path
        self.dataset_type = dataset_type
        self.files = os.listdir(self.data_path+dataset_type)
        self.years = [str(y) for y in years]
        self.used_files = [f for f in self.files if f.split('.')[0].split('_')[1] == str(target_lead)]
        self.used_files = [f for f in self.used_files if f.split('.')[0].split('_')[0][:4] in self.years]        
        self.used_files.sort()
        self.data_list = []
        self.dates = [f.split('.')[0].split('_')[0] for f in self.used_files]
        
        # for f in tqdm(self.used_files):
        #     data = np.load(self.data_path+'/'+self.dataset_type+'/'+f,allow_pickle=True)
        #     self.data_list.append([data['cumul'][0],list(data['instant'])])        
            
    def __len__(self) -> int:
        return len(self.used_files)
            
    def __getitem__(self, raw_idx):        
        # hsr & cumul data stack
        fname = self.used_files[raw_idx]
        hidden_state = np.load(self.data_path+'/'+self.dataset_type+'/'+fname,allow_pickle=True)['hidden']

        return hidden_state, raw_idx

    
class Unet3HiddenStateDataset2nd(torch.utils.data.Dataset):
    """
    """
    def __init__(self, hidden_state_path, file_list,index_list, window_idx_list, dataset_type='train',cluster_lvl=1, cluster_label=0):

        self.data_path = hidden_state_path
        self.dataset_type = dataset_type
        self.used_files = file_list
        self.dataset_type = dataset_type
        self.index_list = index_list
        self.used_index_list = self.index_list[self.index_list[:,1+cluster_lvl]==cluster_label,:]
        self.window_idx_list = window_idx_list
                            
    def __len__(self) -> int:
        return len(self.used_index_list)
            
    def __getitem__(self, raw_idx):        
        # hsr & cumul data stack
        target = self.used_index_list[raw_idx]
        fname = self.used_files[target[0]]
        window = self.window_idx_list[target[1]:target[1]+1]
        hidden_state = np.expand_dims(np.load(self.data_path+'/'+self.dataset_type+'/'+fname,allow_pickle=True)['hidden'],0)
        target_hidden = make_sample_array_sq(hidden_state,window,True)
        
        return target_hidden[0], raw_idx
    

class HiddenStateDataset3rd(torch.utils.data.Dataset):
    """
    """
    def __init__(self, hidden_state_path, file_list,index_list, window_idx_list, dataset_type='train',cluster_lvl=[1], cluster_label=[0]):

        self.data_path = hidden_state_path
        self.dataset_type = dataset_type
        self.used_files = file_list
        self.dataset_type = dataset_type
        self.index_list = index_list
        conditions = [self.index_list[:,1+cluster_lvl[i]]==cluster_label[i] for i in range(0,len(cluster_lvl))]
        conditions = np.stack(conditions,-1)
        self.used_index_list = self.index_list[np.all(conditions,1),:]
        self.window_idx_list = window_idx_list
                            
    def __len__(self) -> int:
        return len(self.used_index_list)
            
    def __getitem__(self, raw_idx):        
        # hsr & cumul data stack
        target = self.used_index_list[raw_idx]
        fname = self.used_files[target[0]]
        window = self.window_idx_list[target[1]:target[1]+1]
        hidden_state = np.expand_dims(np.load(self.data_path+'/'+self.dataset_type+'/'+fname,allow_pickle=True)['hidden'],0)
        target_hidden = make_sample_array_sq(hidden_state,window,True)
        
        return target_hidden[0], raw_idx    
    

class RadarV2HiddenStateDatasetIdx(torch.utils.data.Dataset):
    """
    """
    def __init__(self, hidden_state_path, years, dataset_type:str = 'train',
                ):

        self.data_path = hidden_state_path
        self.dataset_type = dataset_type
        self.files = os.listdir(self.data_path+dataset_type)
        self.years = [str(y) for y in years]
        self.used_files = [f for f in self.files if f.split('.')[0].split('_')[0][:4] in self.years]
        self.used_files.sort()
        self.data_list = []
        
        # for f in tqdm(self.used_files):
        #     data = np.load(self.data_path+'/'+self.dataset_type+'/'+f,allow_pickle=True)
        #     self.data_list.append(data['dates'])        
            
    def __len__(self) -> int:
        return len(self.used_files)
            
    def __getitem__(self, raw_idx):        
        # hsr & cumul data stack
        fname = self.used_files[raw_idx]
        hidden_state = np.load(self.data_path+'/'+self.dataset_type+'/'+fname,allow_pickle=True)['hidden']

        return hidden_state, raw_idx    


class RadarV2HiddenStateDataset2nd(torch.utils.data.Dataset):
    """
    """
    def __init__(self, hidden_state_path, file_list,index_list, window_idx_list, dataset_type='train',cluster_lvl=1, cluster_label=0):

        self.data_path = hidden_state_path
        self.dataset_type = dataset_type
        self.used_files = file_list
        self.dataset_type = dataset_type
        self.index_list = index_list
        self.used_index_list = self.index_list[self.index_list[:,1+cluster_lvl]==cluster_label,:]
        self.window_idx_list = window_idx_list
                            
    def __len__(self) -> int:
        return len(self.used_index_list)
            
    def __getitem__(self, raw_idx):        
        # hsr & cumul data stack
        target = self.used_index_list[raw_idx]
        fname = self.used_files[target[0]]
        window = self.window_idx_list[target[1]:target[1]+1]
        hidden_state = np.expand_dims(np.load(self.data_path+'/'+self.dataset_type+'/'+fname,allow_pickle=True)['hidden'],0)
        target_hidden = make_sample_array_sq(hidden_state,window,True)
        
        return target_hidden[0], raw_idx
    