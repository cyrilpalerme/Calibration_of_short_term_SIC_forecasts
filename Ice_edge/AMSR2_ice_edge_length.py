#!/usr/bin/env python
# coding: utf-8

# In[9]:


import time
import os
import h5py
import datetime
import netCDF4
import numpy as np


# In[10]:


SGE_TASK_ID = 400
#
date_min = "20210101"
date_max = "20221231"
#
paths = {}
paths["AMSR2"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/AMSR2_TOPAZ4_grid/"
paths["UNet"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Training/Land_free_ocean/"
paths["output"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/AMSR2_ice_edge_length/"
#
SIC_thresholds = ["10", "15", "20"]
#
grid_resolution = 12500


# In[11]:


def make_list_dates(date_min, date_max):
    current_date = datetime.datetime.strptime(date_min, "%Y%m%d")
    end_date = datetime.datetime.strptime(date_max, "%Y%m%d")
    list_dates = []
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        list_dates.append(date_str)
        current_date = current_date + datetime.timedelta(days = 1)
    return(list_dates)


# In[12]:


def ice_edge_position(SIE, LSM):
    # LSM => 1 ocean  / 0 land and on the same grid as the SIE
    xdim, ydim = np.shape(SIE)
    nb_neighbors_open_ocean = np.zeros((xdim, ydim))
    for i in range(0, xdim):
        for j in range(0, ydim):
            if (i > 0 and i < xdim-1 and j > 0 and j < ydim-1):
                neighbors_SIE = [SIE[i-1,j], SIE[i+1,j], SIE[i, j-1], SIE[i, j+1]]
                neighbors_ocean = [LSM[i-1,j], LSM[i+1,j], LSM[i, j-1], LSM[i, j+1]]
            elif (i == 0 and j > 0 and j < ydim-1):
                neighbors_SIE = [SIE[i+1,j], SIE[i, j-1], SIE[i, j+1]]
                neighbors_ocean = [LSM[i+1,j], LSM[i, j-1], LSM[i, j+1]]
            elif (i == xdim-1 and j > 0 and j < ydim-1):
                neighbors_SIE = [SIE[i-1,j], SIE[i, j-1], SIE[i, j+1]]
                neighbors_ocean = [LSM[i-1,j], LSM[i, j-1], LSM[i, j+1]]
            elif (i > 0 and i < xdim-1 and j == 0):
                neighbors_SIE = [SIE[i-1,j], SIE[i+1,j], SIE[i, j+1]]
                neighbors_ocean = [LSM[i-1,j], LSM[i+1,j], LSM[i, j+1]]
            elif (i > 0 and i < xdim-1 and j == ydim-1):
                neighbors_SIE = [SIE[i-1,j], SIE[i+1,j], SIE[i, j-1]]
                neighbors_ocean = [LSM[i-1,j], LSM[i+1,j], LSM[i, j-1]]
            elif (i == 0 and j == 0):
                neighbors_SIE = [SIE[i+1,j], SIE[i, j+1]]
                neighbors_ocean = [LSM[i+1,j], LSM[i, j+1]]
            elif (i == 0 and j == ydim-1):
                neighbors_SIE = [SIE[i+1,j], SIE[i, j-1]]
                neighbors_ocean = [LSM[i+1,j], LSM[i, j-1]]
            elif (i == xdim-1 and j == 0):
                neighbors_SIE = [SIE[i-1,j], SIE[i, j+1]]
                neighbors_ocean = [LSM[i-1,j], LSM[i, j+1]]
            elif (i == xdim-1 and j == ydim-1):
                neighbors_SIE = [SIE[i-1,j], SIE[i, j-1]]
                neighbors_ocean = [LSM[i-1,j], LSM[i, j-1]]
            #
            neighbors_SIE = np.array(neighbors_SIE)
            neighbors_ocean = np.array(neighbors_ocean)
            neighbors_open_ocean = np.zeros(len(neighbors_SIE))
            neighbors_open_ocean[np.logical_and(neighbors_SIE == 0, neighbors_ocean == 1)] = 1
            nb_neighbors_open_ocean[i,j] = np.nansum(neighbors_open_ocean)
    ###
    ice_edge = np.logical_and(nb_neighbors_open_ocean >= 1, SIE == 1)
    return(ice_edge)


# In[13]:


def length_sea_ice_edge(ice_edge, spatial_resolution):
    xdim, ydim = np.shape(ice_edge)
    length_sie = np.zeros(np.shape(ice_edge))
    for i in range(0, xdim):
        for j in range(ydim):
            if ice_edge[i,j] == 1:
                if (i > 0 and i < xdim-1 and j > 0 and j < ydim-1):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i-1,j], ice_edge[i+1,j], ice_edge[i,j-1], ice_edge[i,j+1]]))
                elif (i == 0 and j > 0 and j < ydim-1):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i+1,j], ice_edge[i, j-1], ice_edge[i, j+1]]))
                elif (i == xdim-1 and j > 0 and j < ydim-1):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i-1,j], ice_edge[i, j-1], ice_edge[i, j+1]]))
                elif (i > 0 and i < xdim-1 and j == 0):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i-1,j], ice_edge[i+1,j], ice_edge[i, j+1]]))
                elif (i > 0 and i < xdim-1 and j == ydim-1):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i-1,j], ice_edge[i+1,j], ice_edge[i, j-1]]))
                elif (i == 0 and j == 0):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i+1,j], ice_edge[i, j+1]]))
                elif (i == 0 and j == ydim-1):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i+1,j], ice_edge[i, j-1]]))
                elif (i == xdim-1 and j == 0):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i-1,j], ice_edge[i, j+1]]))
                elif (i == xdim-1 and j == ydim-1):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i-1,j], ice_edge[i, j-1]]))
                #
                nb_neighbors_sie = np.array(nb_neighbors_sie)
                if np.sum(nb_neighbors_sie) == 0:
                    length_sie[i,j] = np.sqrt(2) * spatial_resolution
                elif np.sum(nb_neighbors_sie) == 1:
                    length_sie[i,j] = 0.5 * (spatial_resolution + np.sqrt(2) * spatial_resolution)
                elif np.sum(nb_neighbors_sie) >= 2:
                    length_sie[i,j] = spatial_resolution
    #
    sie_length = np.sum(length_sie)
    return(sie_length)


# In[14]:


def calculate_ice_edge_length(date_obs, SIC_thresholds, paths):
    Ice_edge_lengths = {}
    #
    file_UNet = paths["UNet"] + "2021/01/Dataset_20210101.nc"
    nc_UNet = netCDF4.Dataset(file_UNet, "r")
    LSM = nc_UNet.variables["LSM"][:,:]
    nc_UNet.close()
    #
    file_AMSR2 = paths["AMSR2"] + date_obs[0:4] + "/" + date_obs[4:6] + "/" + "SIC_COSI_UNetgrid_" + date_obs + ".nc"
    nc_AMSR2 = netCDF4.Dataset(file_AMSR2, "r")
    SIC = nc_AMSR2.variables["SIC"][:,:]
    nc_AMSR2.close()
    #
    for thresh_SIC in SIC_thresholds:
        SIE = np.zeros(np.shape(SIC))
        SIE[SIC >= int(thresh_SIC)] = 1
        ice_edge = ice_edge_position(SIE, LSM)
        Ice_edge_lengths["SIC" + thresh_SIC] = length_sea_ice_edge(ice_edge, grid_resolution)
    #
    return(Ice_edge_lengths)


# In[15]:


def write_output(Ice_edge_lengths, date_obs, paths, SIC_thresholds):
    path_output = paths["output"] + date_obs[0:4] + "/" + date_obs[4:6] + "/"
    if os.path.isdir(path_output) == False:
        os.system("mkdir -p " + path_output)
    #
    filename_output = path_output + "Ice_edge_lengths_" + date_obs + ".h5"
    hf = h5py.File(filename_output, "w")
    for var in Ice_edge_lengths:
        print(var)
        output_var = "Ice_edge_lengths_" + var
        hf.create_dataset(output_var, data = Ice_edge_lengths[var])
    hf.close()


# In[16]:


t0 = time.time()
list_dates = make_list_dates(date_min, date_max)
date_obs = list_dates[SGE_TASK_ID -1]
Ice_edge_lengths = calculate_ice_edge_length(date_obs, SIC_thresholds, paths)
write_output(Ice_edge_lengths, date_obs, paths, SIC_thresholds)
tf = time.time() - t0
print("Computing time", tf)

