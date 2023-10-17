import numpy as np
import scipy as scp
import pandas as pd
import os
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from pyproj import Proj
from matplotlib import pyplot as plt

from zrdataset_rain import RainDataset, DemoDataset
from hidden_state_dataset import SlicHiddenStateDataset, SlicHiddenStateClusterDatasetV2
from subimage_square import make_subimages
from unet3_model import DenoisingMultiUNet

import dash_bootstrap_components as dbc

from app_constants import *

from mpl_toolkits.basemap import Basemap

def get_latlon():
    
    # load map
    latlon = np.load('dataset/latlon.npy')

    lats = latlon[:, :, 1]
    lons = latlon[:, :, 0]

    m = Basemap(projection = 'merc', lat_0 = 38,  lon_0 = 126,  resolution = 'h', 
                    llcrnrlon=lons.min(), urcrnrlon=lons.max(), llcrnrlat=lats.min(), urcrnrlat=lats.max())
    
    return m, lats, lons 


def load_datasets():
    train_path = preloaded_dataset_path+'train_dataset.pickle'
    valid_path = preloaded_dataset_path+'valid_dataset.pickle'
    train_raw_path = preloaded_dataset_path+'train_dataset_raw.pickle'
    valid_raw_path = preloaded_dataset_path+'valid_dataset_raw.pickle'    
    if os.path.exists(train_path):
        train_dataset = pickle.load(open(train_path, "rb"))
    else:
        train_dataset = DemoDataset(True)
        pickle.dump(train_dataset, open(train_path, "wb"))

    if os.path.exists(valid_path):
        valid_dataset = pickle.load(open(valid_path, "rb"))
    else:        
        valid_dataset = DemoDataset(True)
        pickle.dump(valid_dataset, open(valid_path, "wb"))

    if os.path.exists(train_raw_path):
        train_raw_dataset = pickle.load(open(train_raw_path, "rb"))
    else:        
        train_raw_dataset = DemoDataset(False)
        pickle.dump(train_raw_dataset, open(train_raw_path, "wb"))
        
    if os.path.exists(valid_raw_path):
        valid_raw_dataset = pickle.load(open(valid_raw_path, "rb"))
    else:        
        valid_raw_dataset = DemoDataset(False)
        pickle.dump(valid_raw_dataset, open(valid_raw_path, "wb"))
    
    return train_dataset, valid_dataset, train_raw_dataset, valid_raw_dataset

def load_raw_lat_lon(resolution):
    if resolution == 0.5:
        img_x = 2304 
        img_y = 2880 
    elif resolution == 1.:
        img_x = 1152 
        img_y = 1440     
    
    lat_path = preloaded_dataset_path+f'geo_lat_raw{resolution}.npy'
    lon_path = preloaded_dataset_path+f'geo_lon_raw{resolution}.npy'

    geo_pos = np.array([[y, x] for y in range(0, img_y) for x in range(0, img_x)])
    p = Proj("+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38.0 +lon_0=126.0 +x_0=560000 +y_0=840000 +no_defs +ellps=WGS84 +units=km", reserve_units=True)    

    try:
        geo_lat_raw = np.load(lat_path)
    except:
        geo_lat_raw = np.array([p(x[1], x[0], inverse=True)[1] for x in geo_pos]).reshape(1, img_y, -1)
        np.save(lat_path, geo_lat_raw)    
    
    try:
        geo_lon_raw = np.load(lon_path)
    except:        
        geo_lon_raw = np.array([p(x[1], x[0], inverse=True)[0] for x in geo_pos]).reshape(1, img_y, -1)
        np.save(lon_path, geo_lon_raw)
        
    return torch.Tensor(geo_lat_raw), torch.Tensor(geo_lon_raw)



def load_model():
    dummy_check = True
    sampling = 1
    NUM_CLS = 8
    IMG_DIM = 12
    TIME_DIM = 6

    model = DenoisingMultiUNet(out_channels=NUM_CLS, 
                           img_dim = IMG_DIM+1, 
                           time_dim = TIME_DIM, 
                           use_dropout=False, 
                           use_batchnorm=True, 
                           use_batchnorm_at_first=True,
                           resolution=resolution).to(device)
    
    loaded = torch.load(model_path+'pretrained_best.pt',map_location=device)
    old = loaded['model_state_dict']
    new_state_dict = {}
    for k, v in old.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    model.eval()
    
    print(device)
    
    return model

def load_similarity_items():    
    train_dataset, valid_dataset, train_raw_dataset, valid_raw_dataset = load_datasets()
    similarity_files = {}
    
    return similarity_files

def compute_target_hidden_state(data, bounding_box, model, model_args):    
    x = np.zeros(data.shape)
    return x

def find_similar_input_idx(target_hidden_state, similarity_files, original_dataset, target_lead=1,
                           topk=1):  
    
    org_bb_list = []
    org_img_list = []
    img_date_list = []
    img_inst_list = []
    
    for t in range(topk):
        for idx,item in enumerate(original_dataset.gt_list):
            if (item[-1] == target_lead) and (item[1] not in img_date_list):
                break
        org_img = original_dataset[idx][0]
        org_bb_list.append([(700,900),(800,1100)])
        org_img_list.append(org_img)
        img_date_list.append(item[1])
        img_inst_list.append(item[2])

    return org_img_list, org_img_list, org_bb_list, org_bb_list, img_date_list, img_inst_list

def create_shape(x, y, size=(4,4), color='rgba(39,43,48,10)'):
    shape = [
        {'editable': False,
            'xref': 'x',
            'yref': 'y',
            'layer': 'above',
#            'opacity': 0.2,
            'line': {
                'color': 'black',
                #'width': 1,
                'dash': 'solid'
            },
            'type': 'rect',
            'x0': x,
            'y0': y,
            'x1': x+size[0],
            'y1': y+size[1],
        }
    ]
    
    return shape

def draw_geographic_graph(img, lats, lons, cmap, bounds_labels, bounds_vals, norm, fnames):
    PAD0=0.5
    FONTSIZE=15

    n, _, _ = img.shape
    
    fig_input, ax_input = plt.subplots(figsize=(6,12))  
    m = Basemap(projection = 'merc', lat_0 = 38,  lon_0 = 126,  resolution = 'h', 
                    llcrnrlon=lons.min(), urcrnrlon=lons.max(), llcrnrlat=lats.min(), urcrnrlat=lats.max(),
               ax=ax_input,)

    # Compute interpolated lat, lon, imgs
    lats_scaled = F.interpolate(torch.Tensor(lats).unsqueeze(0).unsqueeze(0), 
                                     (lats.shape[0]//graph_steps, lats.shape[1]//graph_steps),
                                    mode='bilinear',
                                    align_corners=True).squeeze(0).squeeze(0).numpy()
    lons_scaled = F.interpolate(torch.Tensor(lons).unsqueeze(0).unsqueeze(0), 
                                     (lats.shape[0]//graph_steps, lats.shape[1]//graph_steps),
                                    mode='bilinear',
                                    align_corners=True).squeeze(0).squeeze(0).numpy()    
    img_scaled = torch.zeros(img.shape[0], lats.shape[0]//graph_steps, lats.shape[1]//graph_steps)
    for j in range(0,n):
        img_scaled[j,:,:] = F.interpolate(img.unsqueeze(0)[:,j:j+1,:,:], 
                                     (lats.shape[0]//graph_steps, lats.shape[1]//graph_steps),
                                    mode='bilinear',
                                    align_corners=True).squeeze(0)   
    img_scaled = img_scaled.numpy()
    
    for i in tqdm(range(0,n)):
        m.drawcoastlines()
        m.drawcountries()
        m.drawparallels(np.arange(-90.,80.,5.),labels=[1,0,0,0], color='dimgray', dashes=[1,1], fontsize=FONTSIZE, zorder=3)
        m.drawmeridians(np.arange(-180.,180.,5.),labels=[0,0,0,1], color='dimgray', dashes=[1,1], fontsize=FONTSIZE, zorder=3)        
        
        m.pcolormesh(lons_scaled,
                     lats_scaled,
                     np.flipud(img_scaled[i]),
                     latlon=True,
                     cmap=cmap, 
                     norm=norm, 
                     alpha=0.8, 
                     shading='nearest', 
                     zorder=1)
        cbar = m.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), 
                          location='bottom', size='2%', pad=PAD0, extend='both', ticks=bounds_vals)
        cbar.ax.set_xticklabels(labels=bounds_labels, fontsize=FONTSIZE)
        cbar.set_label('Radar(mm/hr)', rotation=360, fontsize=FONTSIZE)
        cbar.ax.tick_params(width=1, length=2, pad=3)
        cbar.outline.set_edgecolor('k')
        cbar.outline.set_linewidth(0.5)
        fig_input.savefig(fnames[i], bbox_inches='tight')

        
def draw_pred_graph(pred, lats, lons, scale_factor, cmap, bounds_labels, bounds_vals, norm, fname):
    PAD0=0.5
    FONTSIZE=15
    
    fig_pred, ax_pred = plt.subplots(figsize=(6,12))  
    m = Basemap(projection = 'merc', lat_0 = 38,  lon_0 = 126,  resolution = 'h', 
                    llcrnrlon=lons.min(), urcrnrlon=lons.max(), llcrnrlat=lats.min(), urcrnrlat=lats.max(),
                   ax=ax_pred)
    
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(-90.,80.,5.),labels=[1,0,0,0], color='dimgray', dashes=[1,1], fontsize=FONTSIZE, zorder=3)
    m.drawmeridians(np.arange(-180.,180.,5.),labels=[0,0,0,1], color='dimgray', dashes=[1,1], fontsize=FONTSIZE, zorder=3)

    # Compute interpolated lat, lon, imgs
    lats_scaled = F.interpolate(torch.Tensor(lats).unsqueeze(0).unsqueeze(0), 
                                     (lats.shape[0]//scale_factor, lats.shape[1]//scale_factor),
                                    mode='bilinear',
                                    align_corners=True).squeeze(0).squeeze(0).numpy()
    lons_scaled = F.interpolate(torch.Tensor(lons).unsqueeze(0).unsqueeze(0), 
                                     (lats.shape[0]//scale_factor, lats.shape[1]//scale_factor),
                                    mode='bilinear',
                                    align_corners=True).squeeze(0).squeeze(0).numpy()    
    
    pred_scaled = F.interpolate(torch.Tensor(pred).unsqueeze(0).unsqueeze(0), 
                                 (lats.shape[0]//scale_factor, lats.shape[1]//scale_factor),
                                mode='nearest-exact',
                                ).squeeze(0).squeeze(0).numpy() 
    
    m.pcolormesh(lons_scaled,
                 lats_scaled,
                 np.flipud(pred_scaled),
                 latlon=True,
                 cmap=cmap, 
                 norm=norm, 
                 alpha=0.8, 
                 shading='nearest', 
                 zorder=1)
    cbar = m.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), 
                      location='bottom', size='2%', pad=PAD0, extend='both', ticks=bounds_vals)
    cbar.ax.set_xticklabels(labels=bounds_labels, fontsize=FONTSIZE)
    cbar.set_label('Class',rotation=360, fontsize=FONTSIZE)
    cbar.ax.tick_params(width=1, length=2, pad=3)
    cbar.outline.set_edgecolor('k')
    cbar.outline.set_linewidth(0.5)
    fig_pred.savefig(fname, bbox_inches='tight')        


def find_nearest(lon_array, x, lat_array, y):
    lon_idx = np.argmin(np.abs(np.min(lon_array,axis=0)-x))
    lat_idx = np.argmin(np.abs(np.min(lat_array,axis=1)-y))
    return [lat_idx,lon_idx]
        
def shape_to_xy(current_shape, common_map, lons, lats):
    # First, need to find the ACTUAL coordinates in the data to use
    # The projection map DISTORTS the raster's shape, so the coordinates
    # on the map do not correspond to that on the raster (see a prediction
    # to see the distortion).

    # 1. Find min/max lon/lat. They correspond to min/max x/y
    lonmin=min(common_map.boundarylons)
    lonmax=max(common_map.boundarylons)
    latmin=min(common_map.boundarylats)
    latmax=max(common_map.boundarylats)

    lonscale = lonmax-lonmin
    latscale = latmax-latmin

    # 2. Scale x & y to find the lon/lat of selected x & y
    x_scale = single_window_limits['x1']-single_window_limits['x0']
    y_scale = single_window_limits['y1']-single_window_limits['y0']

    # Longitude INCREASES as we move right.
    x0 = (current_shape['x0'] - single_window_limits['x0'])*(lonscale/x_scale)+lonmin
    x1 = (current_shape['x1'] - single_window_limits['x0'])*(lonscale/x_scale)+lonmin

    # Latitude DECREASES as we move down.    
    y0 = latmax - (current_shape['y0'] - single_window_limits['y0'])*(latscale/y_scale)
    y1 = latmax - (current_shape['y1'] - single_window_limits['y0'])*(latscale/y_scale)

    # 3. Find the closest coordinates in lon/lat numpy files.
    coord1 = find_nearest(lons,x0,lats,y0)
    coord2 = find_nearest(lons,x0,lats,y1)
    coord3 = find_nearest(lons,x1,lats,y0)
    coord4 = find_nearest(lons,x1,lats,y1)

    new_xs = [coord1[0],coord2[0],coord3[0],coord4[0]]
    new_ys = [coord1[1],coord2[1],coord3[1],coord4[1]]
    
    new_xs.sort()
    new_ys.sort()
    # 4. Choose the tightest range (1st & 3rd values)

    x0 = new_xs[0]
    y0 = new_ys[0]
    x1 = new_xs[2]
    y1 = new_ys[2]        

    bounding_box = [(x0,y0),(x1,y1)]  
    
    return bounding_box

def xy_to_shape(bounding_box, current_shape, common_map, lons, lats):
    [(x0,y0),(x1,y1)] = bounding_box

    lonmin=min(common_map.boundarylons)
    lonmax=max(common_map.boundarylons)
    latmin=min(common_map.boundarylats)
    latmax=max(common_map.boundarylats)    
    
    lonscale = lonmax-lonmin
    latscale = latmax-latmin

    x_scale = single_window_limits['x1']-single_window_limits['x0']
    y_scale = single_window_limits['y1']-single_window_limits['y0']    
    
    # 1. Convert bounding box coordinates into lat/lon
    # 
    lon0 = np.min(lons[:,y0:y0+1],axis=0)
    lat0 = np.min(lats[x0:x0+1,:],axis=1)
    lon1 = np.min(lons[:,y1:y1+1],axis=0)
    lat1 = np.min(lats[x1:x1+1,:],axis=1)    

    # 2. Convert lons and lats into x & y on image
    # Longitude increases as we move right.
    x0 = int((lon0-lonmin)/lonscale*x_scale + single_window_limits['x0'])
    x1 = int((lon1-lonmin)/lonscale*x_scale + single_window_limits['x0'])

    # Latitude decreases as we move down.
    y0 = int((latmax-lat0)/latscale*y_scale + single_window_limits['y0'])
    y1 = int((latmax-lat1)/latscale*y_scale + single_window_limits['y0']) 
        
    # Notice that y0 and y1 do not match with x0 and x1 
    
    return x0, y0, x1, y1

def find_date_in_dataset(target_datetime, target_lead, target_dataset):
    item = None
    for i in tqdm(range(0,len(target_dataset))):
        item_i = target_dataset.gt_list[i]
        if item_i[1] == target_datetime and item_i[-1] == int(target_lead):
            item = item_i
            break       
    
    return i, item

def make_custom_slider(data_datetime, fig):
    sliders=[
        {
            "active": 0,
            "currentvalue": {"prefix": "",
                             "font":{"color": 'black','size':chart_font_size},
                            },
            "len": 0.9,
            
             "font": {
                "color": 'white',
              },            
            "steps": [
                {
                    "args": [
                        [fig.frames[i].name],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "fromcurrent": True,
                        },
                    ],
                    "label": data_datetime[i],
                    "method": "animate",                    
                }
                for i in range(len(fig.frames))
            ],
        }
    ]
    
    return sliders

def make_axis_specs():
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=True,
        linewidth=2,
        linecolor='black',
        gridwidth=1,
        gridcolor='black',
        ticks='outside',
        mirror=True,
        )
    
    yaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=True,
        linewidth=2,
        linecolor='black',  
        gridwidth=1,
        gridcolor='black',        
        ticks='outside',
        mirror=True,
    )
    
    return xaxis, yaxis


# Geograph
# https://plotly.com/python/scatter-plots-on-maps/
# X Array
# https://stackoverflow.com/questions/70319614/how-to-create-a-numpy-array-to-an-xarray-data-array
# https://plotly.com/python/imshow/
# Animation
# https://plotly.com/python/animations/
# Shapes
# https://plotly.com/python/shapes/