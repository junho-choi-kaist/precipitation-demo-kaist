import numpy as np
import scipy as scp
import pandas as pd
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import colors
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA
from torchvision import transforms
import pickle

def denormalize_hsr(tanhed_input):
    tmp = np.arctanh(tanhed_input)*4
    dBz = np.exp(tmp)-0.01
    return dBz # rain

def denormalize_hsr_torch(tanhed_input):
    tmp = torch.atanh(tanhed_input)*4
    dBz = torch.exp(tmp)-0.01
    return dBz # rain

def my_cmap_radar_kma():
    '''
    Usage:
    plt.imshow( ..., cmap=cmap_radar_kma)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap_radar_kma, norm=norm),
                        extend='both',
                        ticks=bounds,
                        label='(mm/hr)',)
    cbar.ax.set_yticklabels([str(x) for x in bounds])
    '''
    # Discrete intervals, some other units
    cmaplist=['#ffffff', '#03c6fe', '#009ef9', '#0047f9', 
                '#01fe04', '#0bb909', '#008b01', '#0a5407', 
                '#fcfa11', '#ffdc1d', '#f5d200', '#e6b700', '#ceab00', 
                '#ff6700', '#ff3400', '#d00708', '#b30305', 
                '#dda4fe', '#ce67fe', '#af2ff8', '#9000e5', 
                '#b4b4e3', '#4c4eaa', '#000392',]
    cmap_radar_kma = (colors.ListedColormap(name='my_cmap_radar_kma', 
                                                  colors=cmaplist, N=len(cmaplist))
                                                 .with_extremes(over='0.25', bad='1', under='1'))
    bounds = [0, 0.01, 0.1, 0.5, 
              1, 2, 3, 4, 
              5, 6, 7, 8, 9, 
              10, 15, 20, 25, 
              30, 40, 50, 60, 
              70, 90, 110]
    bounds_label=['0', '.1', '.5',
                  '1','2','3','4',
                  '5','6','7','8','9',
                  '10','15','20','25',
                  '30','40','50','60',
                  '70','90','110']
    # plt.register_cmap(cmap=cmap_radar_kma)
    norm = colors.BoundaryNorm(bounds, cmap_radar_kma.N)
    # bounds_vals=[0.01, 0.1, 1, 5, 10, 30, 70]
    # bounds_labels=['0', '.1','1','5','10','30','70']
    bounds_vals=[0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 70, 110]
    bounds_labels=['0', '.1', '.5', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '20', '30', '50', '70', '110']
    return cmap_radar_kma, bounds_vals, bounds_labels, norm
    # return cmap_radar_kma, bounds[1:], bounds_label, norm
    
def my_cmap_radar_kma_pred(n_class):
    '''
    Usage:
    plt.imshow( ..., cmap=cmap_radar_kma)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap_radar_kma, norm=norm),
                        extend='both',
                        ticks=bounds,
                        label='(mm/hr)',)
    cbar.ax.set_yticklabels([str(x) for x in bounds])
    '''
    # Discrete intervals, some other units
    cmaplist=['#ffffff', '#03c6fe', '#009ef9', '#0047f9', 
                '#01fe04', '#0bb909', '#008b01', '#0a5407', 
                '#fcfa11', '#ffdc1d', '#f5d200', '#e6b700', '#ceab00', 
                '#ff6700', '#ff3400', '#d00708', '#b30305', 
                '#dda4fe', '#ce67fe', '#af2ff8', '#9000e5', 
                '#b4b4e3', '#4c4eaa', '#000392',]
    cmaplist = cmaplist[::2][:n_class]
    cmap_radar_kma = (colors.ListedColormap(name='my_cmap_radar_kma', 
                                                  colors=cmaplist, N=len(cmaplist))
                                                 .with_extremes(over='0.25', bad='1', under='1'))
    bounds = [0]+[int(i+1) for i in range(0,n_class)]
    bounds_label=[0]+[str(int(i+1)) for i in range(0,n_class)]
    norm = colors.BoundaryNorm(bounds, cmap_radar_kma.N)
    bounds_vals= bounds[1:]
    bounds_labels= bounds_label[1:]
    return cmap_radar_kma, bounds_vals, bounds_labels, norm    
    
    
def my_cmap_radar_kma_px():
    '''
    '''
    # Discrete intervals, some other units
    cmaplist=['#ffffff', '#03c6fe', '#009ef9', '#0047f9', 
                '#01fe04', '#0bb909', '#008b01', '#0a5407', 
                '#fcfa11', '#ffdc1d', '#f5d200', '#e6b700', '#ceab00', 
                '#ff6700', '#ff3400', '#d00708', '#b30305', 
                '#dda4fe', '#ce67fe', '#af2ff8', '#9000e5', 
                '#b4b4e3', '#4c4eaa', '#000392',]
    bounds = [0, 0.01, 0.1, 0.5, 
              1, 2, 3, 4, 
              5, 6, 7, 8, 9, 
              10, 15, 20, 25, 
              30, 40, 50, 60, 
              70, 90, 110]
    bounds_label=['0', '.1', '.5',
                  '1','2','3','4',
                  '5','6','7','8','9',
                  '10','15','20','25',
                  '30','40','50','60',
                  '70','90','110']
    # plt.register_cmap(cmap=cmap_radar_kma)
    norm = colors.BoundaryNorm(bounds, len(cmaplist))
    # bounds_vals=[0.01, 0.1, 1, 5, 10, 30, 70]
    # bounds_labels=['0', '.1','1','5','10','30','70']
    bounds_vals=[0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 70, 110]
    bounds_labels=['0', '.1', '.5', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '20', '30', '50', '70', '110']
    return cmaplist, bounds_vals, bounds_labels, norm    
    


# rain = denormalize_hsr(radar_history).numpy()
# cmap_radar_kma, bounds_vals, bounds_label, norm = my_cmap_radar_kma()

# m.pcolormesh(lons,lats, rain[channel], latlon=True, cmap=cmap_radar_kma, norm=norm, alpha=ALPHA_KMA, shading='nearest')
# cbar = m.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap_radar_kma, norm=norm), 
#                   ax=plt_axis[i], location='bottom', size='2%', pad=PAD0, extend='both', ticks=bounds_vals)
# cbar.ax.set_xticklabels(labels=bounds_label, fontsize=FONTSIZE_SMALL)
# cbar.set_label('Radar(mm/hr)', labelpad=-40, x=0, rotation=360, fontsize=FONTSIZE_SMALL)
# cbar.ax.tick_params(width=1, length=2, pad=3)
# cbar.outline.set_edgecolor('k')
#cbar.outline.set_linewidth(0.5)