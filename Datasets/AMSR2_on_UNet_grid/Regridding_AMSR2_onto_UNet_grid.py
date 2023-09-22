#!/usr/bin/env python
# coding: utf-8

# In[229]:


import os
import time
import scipy
import pyproj
import netCDF4
import datetime
import numpy as np


# # Constants

# In[230]:


SGE_TASK_ID = 2013
yyyy = str(SGE_TASK_ID)
#
paths = {}
paths["AMSR2"] = "/lustre/storeB/project/copernicus/cosi/WP2/SIC/v0.1/"
paths["UNet"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Training/Land_free_ocean/"
paths["output"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/AMSR2_TOPAZ4_grid/"
#
date_min = yyyy + "0101"
date_max = yyyy + "1231"
#
crs = {}
#crs["latlon"] = pyproj.CRS.from_proj4("+proj=latlon")
crs["AMSR2"] = pyproj.CRS.from_proj4("+ellps=WGS84 +lat_0=90 +lon_0=0 +no_defs=None +proj=laea +type=crs +units=m +x_0=0 +y_0=0")
crs["TOPAZ"] = pyproj.CRS.from_proj4("+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere")
#


# In[231]:


def make_list_dates(date_min, date_max):
    current_date = datetime.datetime.strptime(date_min, "%Y%m%d")
    end_date = datetime.datetime.strptime(date_max, "%Y%m%d")
    list_dates = []
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        list_dates.append(date_str)
        current_date = current_date + datetime.timedelta(days = 1)
    return(list_dates)


# In[232]:


def load_UNet_grid(paths, forecast_start_date = "20210101"):
    file_UNet = paths["UNet"] + forecast_start_date[0:4] + "/" + forecast_start_date[4:6] + "/" + "Dataset_" + forecast_start_date + ".nc"
    #
    nc = netCDF4.Dataset(file_UNet, "r")
    UNet = {}
    UNet["x"] = nc.variables["x"][:]
    UNet["y"] = nc.variables["y"][:]
    UNet["lat"] = nc.variables["lat"][:,:]
    UNet["lon"] = nc.variables["lon"][:,:]
    UNet["LSM"] = nc.variables["LSM"][:,:]
    nc.close()
    #
    return(UNet)


# In[233]:


def load_AMSR2_data(paths, date_task):
    date_end = (datetime.datetime.strptime(date_task, "%Y%m%d") + datetime.timedelta(days = 1)).strftime("%Y%m%d")
    date_str = date_task + "0000" + "-" + date_end + "0000" 
    file_AMSR2 = paths["AMSR2"] + date_task[0:4] + "/" + date_task[4:6] + "/" + "sic_cosi-5km_" + date_str + ".nc"
    #
    nc = netCDF4.Dataset(file_AMSR2, "r")
    AMSR2 = {}
    AMSR2["x"] = nc.variables["xc"][:] * 1000
    AMSR2["y"] = nc.variables["yc"][:] * 1000
    AMSR2["SIC"] = nc.variables["ice_conc"][0,:,:]
    AMSR2["total_standard_uncertainty"] = nc.variables["total_standard_uncertainty"][0,:,:]
    nc.close()
    #
    return(AMSR2)


# In[234]:


def nearest_neighbor_indexes(x_input, y_input, x_output, y_output):
    x_input = np.expand_dims(x_input, axis = 1)
    y_input = np.expand_dims(y_input, axis = 1)
    x_output = np.expand_dims(x_output, axis = 1)
    y_output = np.expand_dims(y_output, axis = 1)
    #
    coord_input = np.concatenate((x_input, y_input), axis = 1)
    coord_output = np.concatenate((x_output, y_output), axis = 1)
    #
    tree = scipy.spatial.KDTree(coord_input)
    dist, idx = tree.query(coord_output)
    #
    return(idx)


# In[235]:


def regridding_AMSR2(date_task, list_dates, paths, crs):
    #
    AMSR2_regrid = {}
    #
    transform_AMSR2_to_TOPAZ = pyproj.Transformer.from_crs(crs["AMSR2"], crs["TOPAZ"], always_xy = True)
    #
    UNet = load_UNet_grid(paths)
    xx_UNet, yy_UNet = np.meshgrid(UNet["x"], UNet["y"])
    xx_UNet_flat = np.ndarray.flatten(xx_UNet)
    yy_UNet_flat = np.ndarray.flatten(yy_UNet)
    #
    AMSR2 = load_AMSR2_data(paths, date_task)
    xx_AMSR2, yy_AMSR2 = np.meshgrid(AMSR2["x"], AMSR2["y"])
    xx_SIC_TOPAZproj, yy_SIC_TOPAZproj = transform_AMSR2_to_TOPAZ.transform(xx_AMSR2, yy_AMSR2)
    SIC_AMSR2 = np.ndarray.flatten(AMSR2["SIC"])
    total_uncertainty_AMSR2 = np.ndarray.flatten(AMSR2["total_standard_uncertainty"])
    #
    idx_fill_values = np.logical_or(SIC_AMSR2 < 0, total_uncertainty_AMSR2 < 0)
    xx_flat_SIC_TOPAZproj = np.ndarray.flatten(xx_SIC_TOPAZproj)[idx_fill_values == False]
    yy_flat_SIC_TOPAZproj = np.ndarray.flatten(yy_SIC_TOPAZproj)[idx_fill_values == False]
    SIC_AMSR2 = SIC_AMSR2[idx_fill_values == False] 
    total_uncertainty_AMSR2 = total_uncertainty_AMSR2[idx_fill_values == False] 
    #
    inter_idx = nearest_neighbor_indexes(xx_flat_SIC_TOPAZproj, yy_flat_SIC_TOPAZproj, xx_UNet_flat, yy_UNet_flat)
    #
    AMSR2_regrid["x"] = np.copy(UNet["x"])
    AMSR2_regrid["y"] = np.copy(UNet["y"])
    AMSR2_regrid["lat"] = np.copy(UNet["lat"])
    AMSR2_regrid["lon"] = np.copy(UNet["lon"])
    #
    SIC_interp = SIC_AMSR2[inter_idx]
    total_uncertainty_interp = total_uncertainty_AMSR2[inter_idx]
    AMSR2_regrid["SIC"] = np.reshape(SIC_interp, (len(UNet["y"]), len(UNet["x"])), order = "C")
    AMSR2_regrid["total_uncertainty"] = np.reshape(total_uncertainty_interp, (len(UNet["y"]), len(UNet["x"])), order = "C")
    AMSR2_regrid["SIC"][UNet["LSM"] == 0] = 0
    AMSR2_regrid["total_uncertainty"][UNet["LSM"] == 0] = 0
    #
    return(AMSR2_regrid) 


# In[236]:


def write_netcdf(date_task, AMSR2_regrid, paths):
    path_output = paths["output"] + date_task[0:4] + "/" + date_task[4:6] + "/" 
    if os.path.exists(path_output) == False:
        os.system("mkdir -p " + path_output)    
    output_filename = path_output + "SIC_COSI_UNetgrid_" + date_task + ".nc"
    if os.path.isfile(output_filename):
        os.system("rm " + output_filename)
    #
    output_netcdf = netCDF4.Dataset(output_filename, "w", format = "NETCDF4")
    #
    x = output_netcdf.createDimension("x", len(AMSR2_regrid["x"]))
    y = output_netcdf.createDimension("y", len(AMSR2_regrid["y"]))
    #
    xc = output_netcdf.createVariable("xc", "d", ("x"))
    yc = output_netcdf.createVariable("yc", "d", ("y"))
    lat = output_netcdf.createVariable("lat", "d", ("y","x"))
    lon = output_netcdf.createVariable("lon", "d", ("y","x"))
    SIC = output_netcdf.createVariable("SIC", "d", ("y","x"))
    total_uncertainty = output_netcdf.createVariable("total_uncertainty", "d", ("y","x"))
    #
    xc.units = "m"
    yc.units = "m"
    lat.units = "degree"
    lon.units = "degree"
    SIC.units = "Sea ice concentration from AMSR2 observations (%)"
    total_uncertainty.units = "total standard uncertainty from AMSR2 observations (%)"
    #
    xc[:] = AMSR2_regrid["x"] 
    yc[:] = AMSR2_regrid["y"]
    lat[:,:] = AMSR2_regrid["lat"] 
    lon[:,:] = AMSR2_regrid["lon"]
    SIC[:,:] = AMSR2_regrid["SIC"]
    total_uncertainty[:,:] = AMSR2_regrid["total_uncertainty"]
    #
    output_netcdf.description = "+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere"
    output_netcdf.close()


# In[237]:


t0 = time.time()
list_dates = make_list_dates(date_min, date_max)
for date_task in list_dates:
    print(date_task, time.time() - t0)
    try:
        AMSR2_regrid = regridding_AMSR2(date_task, list_dates, paths, crs)
        write_netcdf(date_task, AMSR2_regrid, paths)
    except:
        pass

