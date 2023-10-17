import numpy as np
import scipy as scp
import pandas as pd
import os
import torch
import pickle
import dash_bootstrap_components as dbc

#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = 'cpu'

util_path = './'
num_workers = 12
cls_thres = [0.1,1,5,10,20,25,30]
leadtime_interval = 60
num_leadtime = 6
batch_size = 200
resolution = 1
lr = 0.00005

instant_path = '/data1/weather/unet3processed/instant/'
cumul_path = '/data1/weather/unet3processed/cumul/'
missing_file_path = '/data1/weather/unet3processed/missing_dates.npz'
preloaded_dataset_path = 'dataset/'
model_path = 'model/'

hidden_state_path = '/data1/weather/hsrslic_combined/unet3/'
cluster_model_path = 'clusters/'
cluster_hidden_size = 'small'
hidden_lead = 1

train_yr_list = [2018,2019,2020]
valid_yr_list = [2018,2019,2020,2021]
target_yr_list = [2019,2020]
view_yr_start = 2019

# Theme-background pairs
theme_dict = {"cyborg":[dbc.themes.CYBORG, "dimgray", "white", ],
              "darkly":[dbc.themes.DARKLY, "dimgray", "white", ],
              "bootstrap":[dbc.themes.BOOTSTRAP, "white", "black", ],
}

step = 32
window_s = ((1+8)*step,(1+8)*step)
window_m = ((10+8)*step,(10+8)*step)
window_l = ((37+8)*step,(28+8)*step)
window = {}
window['small'] = window_s
window['mid'] = window_m
window['large'] = window_l

target_topk = 3

dash_margin = {"margin-left": "15px","margin-right": "5px","margin-top": "15px", "margin-bottom": "5px"}

single_window_size = {'height':700,
                     'width':470}

single_margin = {"b": 15, "r": 5, "l": 0, "pad": 0}
pred_margin = {"b": 15, "r": 5, "l": 5, "pad": 0}
chart_margin = {"b": 5, "r": 5, "l": 5, "pad": 0}

single_window_limits = {'x0':68,
                       'y0':10,
                       'x1':531,
                       'y1':543,
                      }

single_window_ratio = {'x':(531-68)/1152, 
                       'y':(543-10)/1440,} 

graph_steps = 6
pred_pad = 20

# Fonts
header1_style = {'font-family': 'ui-san-serif',
                 'font-weight': 'bold',
                 'font-size': '28px',
                 'text-align': 'left',}

header2_style = {'font-family': 'ui-san-serif',
                 'font-weight': 'bold',
                 'font-size': '18px',
                 'text-align': 'left',}

header3_style = {'font-family': 'ui-san-serif',
                 'font-size': '18px',
                 'text-align': 'left',}

header4_style = {'font-family': 'ui-san-serif',
                 'font-size': '18px',
                 'text-align': 'left',}

item_style = {'font-family': 'ui-san-serif',
              'font-size': '18px',
              'text-align': 'left',}

dropdown_style = {'font-family': 'ui-san-serif',
                  'font-size': '18px',
                  'text-align': 'left',}

for key in header4_style.keys():
    dash_margin[key] = header4_style[key]

chart_font_size = 18

date_form = "{0}년 {1}월 {2}일 {3}:{4}"