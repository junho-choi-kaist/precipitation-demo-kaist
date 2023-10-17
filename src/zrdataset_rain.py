from itertools import chain
import torch
import os
import gzip
import numpy as np
from datetime import datetime, timedelta
from pyproj import Proj
import glob

from tqdm import tqdm

import torch.nn.functional as F

def zr_relation(radar):
    x = torch.FloatTensor(10**(radar*0.1))
    return torch.FloatTensor((x/148)**(100/159))
    
class RainDataset(torch.utils.data.Dataset):
    """
        Args:
            instant_data_path (str) : 순간강수 HSR 데이터셋 경로
            cumul_data_path (str) : 누적강수 HSR 데이터셋 경로
            util_path (str) : 위경도 정보 (geo_lon.py, geo_lat.py) 데이터 경로
            years (list) : 데이터 년도의 list
            cls_thres (list) : 강수량의 class를 구분할 경계선
            num_leadtime (int) : leadtime의 개수
            leadtime_interval (int) : 각 leadtime 사이의 시간 간격
            input_dim (int) : 입력으로 들어갈 순간강수 이미지 개수
            dataset_type (str) : 데이터셋의 용도 (train or valid or test)
            sampling (int) : sampling ratio (0 ~ 1)
            normalize_har (bool) : 순간강수 데이터 normalize 유무
    """
    def __init__(self, instant_data_path : str, cumul_data_path :str, missing_file_path:str = None, util_path:str = '../_util_data/', 
                 years:list = [2018, 2019, 2021, 2022], cls_thres:list = [1, 10], resolution:int = 1,
                 num_leadtime:int = 6, leadtime_interval:int = 60, input_dim:int = 7, 
                 dataset_type:str = 'train', sampling:int = 1.0,
                 normalize_hsr = False, month_list = None,
                 exclude_all_null = True, use_gt1 = True,
                ):
        assert type(input_dim) == int
        assert 0.0 <= sampling <= 1.0
        assert dataset_type in ['train', 'valid', 'test']
        assert len(cls_thres) >= 0
        assert leadtime_interval > 0 and num_leadtime > 0
        assert resolution == 1. or resolution == 0.5
        
        self.cls_thres = cls_thres
        self.input_dim = input_dim
        self.instant_data_path = instant_data_path
        self.cumul_data_path = cumul_data_path
        self.leadtime_interval = timedelta(hours=leadtime_interval/60)
        self.normalize_hsr = normalize_hsr
        self.resolution = resolution
        if month_list is None:
            self.month_list = [i for i in range(1,13)]    
        else:
            self.month_list = month_list
        self.data_interval = timedelta(hours=10/60)
        self.hour_interval = timedelta(hours=60/60)
        self.dataset_type = dataset_type
        self.num_leadtime = num_leadtime
        self.sampling = sampling
        self.exclude_all_null = exclude_all_null
        self.use_gt1 = use_gt1
        
        if missing_file_path is None:
            self.missing_file_path = None
            self.cumul_missing_file = None
            self.instant_missing_file = None
        else:
            self.missing_file_path = missing_file_path
            self.cumul_missing_file = np.load(missing_file_path,allow_pickle=True)['cumul']
            self.instant_missing_file = np.load(missing_file_path,allow_pickle=True)['instant']
        
        print(f"*** Load {dataset_type} dataset ***")
        self.instant_file_list = []
        for yr in years:
            if os.path.exists(f'{self.instant_data_path}/{yr}'):
                file_names = os.listdir(f'{self.instant_data_path}/{yr}')
                self.instant_file_list += file_names
            
        print(f"*** Instant data length: {len(self.instant_file_list)} ***")
        
        self.cumul_file_list = []
        for yr in years:
            if os.path.exists(f'{self.cumul_data_path}/{yr}'):
                file_names = os.listdir(f'{self.cumul_data_path}/{yr}')
                self.cumul_file_list += file_names
            
        print(f"*** Cumulative data length: {len(self.cumul_file_list)} ***")    
        
        # Find dates for instant and cumulative data
        instant_date_set = set()
        for file_name in self.instant_file_list:
            file_date = file_name.split('_')[0]
            instant_date_set.add(file_date)

        
        cumul_date_set = set()
        for file_name in self.cumul_file_list:
            file_date = file_name.split('_')[0]
            cumul_date_set.add(file_date)
     
        
        all_date_set = set()
        for d in instant_date_set:
            all_date_set.add(d)
        for d in cumul_date_set:
            all_date_set.add(d)
        self.instant_date_list = list(instant_date_set)
        self.instant_date_list.sort()        
        self.cumul_date_list = list(cumul_date_set)
        self.cumul_date_list.sort()           
        self.all_date_list = list(all_date_set)
        self.all_date_list.sort()
        
        # For each date in instant data, check if (a) there is a matching
        # cumulative data, and (b) if it is possible to get a history of inputs
        
        self.gt_date_list = set()
        self.gt1_date_list = set()
        gt_dict = {}
        for datestr in tqdm(self.all_date_list):
            # Cannot use date if it is not in cumulative data.
            year = datestr[:4]
            if not os.path.exists(f'{self.cumul_data_path}/{year}/{datestr}_cumul.npz'):
                continue      
            
            # Cannot use date if the dataset is training and the date does not end in 00.
            if (self.dataset_type == 'train') and (datestr[-2:] != '00'):
                continue
                
            # Cannot use date if historical data cannot be found for lead time intervals.
            current_date = datetime.strptime(datestr, "%Y%m%d%H%M")
            
            for i in range(1, self.num_leadtime+1):
                is_input_files_exist = True
                check_date = current_date - self.leadtime_interval*i
                check_datestr = check_date.strftime('%Y%m%d%H%M')
                check_year = check_datestr[:4]
                
                # No cumulative data for chosen date.
                if not os.path.exists(f'{self.cumul_data_path}/{check_year}/{check_datestr}_cumul.npz'):
                    continue
                
                # Not enough instant data.
                date_history = []
                for j in range(0,self.input_dim):
                    new_date = check_date - self.data_interval*(self.input_dim-1-j)
                    new_datestr = new_date.strftime('%Y%m%d%H%M')
                    new_year = new_datestr[:4]
                    if not os.path.exists(f'{self.instant_data_path}/{new_year}/{new_datestr}_instant.npz'):
                        is_input_files_exist = False
                        break
                    else:
                        date_history.append(new_datestr)
                
                if is_input_files_exist:
                    # Target date, historical cumulative data, historical instant data, leadtime
                    self.gt_date_list.add(datestr)
                    if datestr not in gt_dict.keys():
                        gt_dict[datestr] = []
                    gt_dict[datestr].append([datestr, check_datestr, date_history,i]) 
                    if i == 1:
                        self.gt1_date_list.add(datestr)
        
        self.gt_date_list = list(self.gt_date_list)
        self.gt_date_list.sort()
        self.gt1_date_list = list(self.gt1_date_list)
        self.gt1_date_list.sort()
                    
        # If the dataset type is not training, perform sampling
        self.gt_list = []
        if self.dataset_type == 'train':
            for gt_date in self.gt_date_list:
                for gt in gt_dict[gt_date]:
                    self.gt_list.append(gt)    
        if self.dataset_type != 'train':
            jump = 10//(int)(self.sampling*10)
            self.gt1_date_list = self.gt1_date_list[0::jump]
            self.gt_date_list = self.gt_date_list[0::jump]
           
            if self.use_gt1:
                for gt_date in self.gt1_date_list:
                    for gt in gt_dict[gt_date]:
                        self.gt_list.append(gt)
            else:
                for gt_date in self.gt_date_list:
                    for gt in gt_dict[gt_date]:
                        self.gt_list.append(gt)    

        # Filter out gt where either instant data or cumulative data is missing:
        if self.missing_file_path is not None:
            final_gt_list = []
            for gt_data in tqdm(self.gt_list):
                gt, cumul, history, lead = gt_data
                if gt in self.cumul_missing_file:
                    continue
                if cumul in self.cumul_missing_file:
                    continue
                instant_is_available = True
                for hist in history:
                    if hist in self.instant_missing_file:
                        instant_is_available = False
                        break
                if instant_is_available:
                    final_gt_list.append(gt_data)        
            self.gt_list = final_gt_list            
            
        print("\t# of Ground Truths:"+str(len(self.gt_list)))
        
        if self.resolution == 0.5:
            self.img_x = 2304 
            self.img_y = 2880 
        elif self.resolution == 1.:
            self.img_x = 1152 
            self.img_y = 1440 
        
        try:
            self.geo_lon = np.load(util_path + "geo_lon"+str(self.resolution)+".npy")
            self.geo_lat = np.load(util_path + "geo_lat"+str(self.resolution)+".npy")
        except:
            print("\tNeither latitude nor longitude file don't exist")
            geo_pos = np.array([[y, x] for y in range(0, self.img_y) for x in range(0, self.img_x)])
            p = Proj("+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38.0 +lon_0=126.0 +x_0=560000 +y_0=840000 +no_defs +ellps=WGS84 +units=km", reserve_units=True)
            self.geo_lon = np.array([p(x[1], x[0], inverse=True)[0] for x in geo_pos]).reshape(1, self.img_y, -1)
            self.geo_lat = np.array([p(x[1], x[0], inverse=True)[1] for x in geo_pos]).reshape(1, self.img_y, -1)
            self.geo_lon /= np.max(self.geo_lon)
            self.geo_lat /= np.max(self.geo_lat)
            
            np.save(util_path + "geo_lon"+str(self.resolution)+".npy", self.geo_lon)
            np.save(util_path + "geo_lat"+str(self.resolution)+".npy", self.geo_lat)

        self.zero_threshold = zr_relation(np.array([-250]))
        self.zero_threshold_hsr = torch.tanh(torch.log(self.zero_threshold+0.01)/4)            
            
    def __len__(self) -> int:
        return len(self.gt_list)
            
    def __getitem__(self, raw_idx):        
        # hsr & cumul data stack
        
        target_date, cumul_date, date_history, lead = self.gt_list[raw_idx]
        input_data = []
        for inst_date in date_history:
            yr = inst_date[:4]
            data = np.load(f'{self.instant_data_path}/{yr}/{inst_date}_instant.npz',allow_pickle=True)
            if self.normalize_hsr:
                input_data.append(data['data_normal'])
            else:
                input_data.append(data['data'])
        
        yr = cumul_date[:4]
        data = np.load(f'{self.cumul_data_path}/{yr}/{cumul_date}_cumul.npz',allow_pickle=True)
        if self.normalize_hsr:
            input_data.append(data['data_normal'])
        else:
            input_data.append(data['data'])
        
        input_data = np.stack(input_data,axis=0)
        
        t_hour = np.full((self.img_y, self.img_x), int(target_date[8:10])/24).reshape(1, self.img_y, -1)
        t_day = np.full((self.img_y, self.img_x), int(target_date[6:8])/31).reshape(1, self.img_y, -1)
        t_month = np.full((self.img_y, self.img_x), int(target_date[4:6])/12).reshape(1, self.img_y, -1)      
        
        input_data = torch.FloatTensor(np.concatenate((input_data, t_hour, t_day, t_month, self.geo_lat, self.geo_lon), axis = 0))
        
        # GT
        yr = target_date[:4]
        data = np.load(f'{self.cumul_data_path}/{yr}/{target_date}_cumul.npz',allow_pickle=True)
        # if self.normalize_hsr:
        #     gt = data['data_normal']
        # else:
        gt = data['data']
        
        if self.exclude_all_null is False:
            mask = (data['data'] >= -250)
        else:
            mask = (gt >= 0)
        
        # classification 정답
        gt_cls = torch.zeros_like(torch.Tensor(gt))
        cls_idx = 1
        for ct in self.cls_thres:
            gt_cls[gt >= ct] = cls_idx
            cls_idx += 1

        lead_cls = torch.full(gt.shape, (lead-1))

        return input_data, gt_cls, gt, mask, (lead-1), lead_cls
    
    

class DemoDataset(torch.utils.data.Dataset):
    """
        Used for demo only.
    """
    def __init__(self, normalize_hsr=True):
        import pickle
        self.normalize_hsr = normalize_hsr
        self.normalize_tag = 'normalized' if self.normalize_hsr else 'unnormalized'
        self.demo_files = os.listdir('demo_files')
        self.demo_files.sort()
        self.demo_files = [d for d in self.demo_files if self.normalize_tag in d]
        self.demo_files = ['demo_files/'+d for d in self.demo_files]
            
        self.gt_list = []
        for d in self.demo_files:
            gt_spec = pickle.load(open(d, "rb"))['gt_spec']
            self.gt_list.append(gt_spec)
            
        self.input_dim = len(self.gt_list[0][2])

                
    def __len__(self) -> int:
        return len(self.gt_list)
            
    def __getitem__(self, raw_idx):        
        # input_data, gt_cls, gt, mask, (lead-1), lead_cls  
        import pickle
        d = self.demo_files[raw_idx]
        return pickle.load(open(d, "rb"))['data']
        
        


