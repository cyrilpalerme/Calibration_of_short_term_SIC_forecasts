#!/usr/bin/env python
# coding: utf-8

# In[161]:


import os
import time
import scipy
import pyproj
import netCDF4
import datetime
import numpy as np


# In[162]:


date_min = "20201201"
date_max = "20230131"
#
crs = {}
crs["ice_charts"] = pyproj.CRS.from_proj4("+proj=stere lon_0=0.0 lat_ts=90.0 lat_0=90.0 a=6371000.0 b=6371000.0")
crs["UNet"] = pyproj.CRS.from_proj4("+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere")
#
paths = {}
paths["ice_charts"] = "/lustre/storeB/project/copernicus/sea_ice/SIW-METNO-ARC-SEAICE_HR-OBS/"
paths["UNet"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Training/Land_free_ocean/"
paths["output"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Ice_charts_UNet_grid/"


# In[163]:


def make_list_dates(date_min, date_max):
    current_date = datetime.datetime.strptime(date_min, "%Y%m%d")
    end_date = datetime.datetime.strptime(date_max, "%Y%m%d")
    list_dates = []
    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        list_dates.append(date_str)
        current_date = current_date + datetime.timedelta(days = 1)
    return(list_dates)


# # Padding function (make_padding)
# 
#     x and y must be vectors (can be latitude / longitude if the data are on a regular grid)  
#     field must be either a 2D array (y, x) or a 3D array (time, y, x)

# In[164]:


def make_padding(x, y, field):
    dx = x[1] - x[0]
    x_extent = np.pad(x, (1, 1), constant_values = np.nan)    
    x_extent[0] = x_extent[1] - dx
    x_extent[-1] = x_extent[-2] + dx
    #
    dy = y[1] - y[0]
    y_extent = np.pad(y, (1, 1), constant_values = np.nan)
    y_extent[0] = y_extent[1] - dy
    y_extent[-1] = y_extent[-2] + dy
    #
    if field.ndim == 2:
        field_extent = np.pad(field, (1,1), constant_values = np.nan)
    elif field.ndim == 3:
        time_dim = len(field[:,0,0])
        field_extent = np.full((time_dim, len(y_extent), len(x_extent)), np.nan)
        #
        for t in range(0, time_dim):
            field_extent[t,:,:] = np.pad(field[t,:,:], (1,1), constant_values = np.nan)
    #
    return(x_extent, y_extent, field_extent)


# # Regridding functions (nearest_neighbor_indexes and nearest_neighbor_interp)
#     xx_input and yy_input must be 2D arrays
#     x_output and y_output must be vectors  
#     field must be either a 2D array with dimensions (y, x) or a 3D array with dimensions (time, y, x) 
#     invalid_values = fill value to replace by 0. Land is therefore considered as open ocean.

# In[165]:


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


# In[166]:


def extract_idx(Data_ice_charts, UNet_coordinates, crs = crs):
    transform_ice_charts_to_UNet = pyproj.Transformer.from_crs(crs["ice_charts"], crs["UNet"], always_xy = True)
    #
    xx_ice_charts, yy_ice_charts = np.meshgrid(Data_ice_charts["x"], Data_ice_charts["y"])
    xx_UNet, yy_UNet = np.meshgrid(UNet_coordinates["x"], UNet_coordinates["y"])
    #
    xx_ice_charts_UNetproj, yy_ice_charts_UNetproj = transform_ice_charts_to_UNet.transform(xx_ice_charts, yy_ice_charts)
    #
    xx_ice_charts_UNetproj_flat = np.ndarray.flatten(xx_ice_charts_UNetproj)
    yy_ice_charts_UNetproj_flat = np.ndarray.flatten(yy_ice_charts_UNetproj)
    xx_UNet_flat = np.ndarray.flatten(xx_UNet)
    yy_UNet_flat = np.ndarray.flatten(yy_UNet)
    #
    idx = {}
    idx["ice_charts_to_UNet"] = nearest_neighbor_indexes(xx_ice_charts_UNetproj_flat, yy_ice_charts_UNetproj_flat, xx_UNet_flat, yy_UNet_flat)
    #
    return(idx)


# In[167]:


def load_ice_chart_data(filename):
    Data = {}
    nc = netCDF4.Dataset(filename, "r")
    x = nc.variables["xc"][:]
    y = nc.variables["yc"][:]
    SIC = nc.variables["ice_concentration"][0,:,:]
    Data["x"], Data["y"], Data["SIC"] = make_padding(x, y, SIC)
    LSM_field = np.zeros(np.shape(Data["SIC"]))
    LSM_field[Data["SIC"] > -90] = 1
    Data["LSM"] = np.copy(LSM_field)
    nc.close()
    return(Data)


# In[168]:


def load_UNet_coordinates(date_task = "20220101"):
    Data = {}
    filename = paths["UNet"] + date_task[0:4] + "/" + date_task[4:6] + "/" + "Dataset_" + date_task + ".nc"
    nc = netCDF4.Dataset(filename, "r")
    Data["x"] = nc.variables["x"][:]
    Data["y"] = nc.variables["y"][:]
    Data["lat"] = nc.variables["lat"][:,:]
    Data["lon"] = nc.variables["lon"][:,:]
    Data["LSM"] = nc.variables["LSM"][:,:]
    nc.close()
    return(Data)


# In[169]:


def nearest_neighbor_interp(idx, Data_ice_charts, UNet_coordinates):
    Data_UNet_grid = {}
    #
    SIC_flat = np.ndarray.flatten(Data_ice_charts["SIC"])
    LSM_flat = np.ndarray.flatten(Data_ice_charts["LSM"])
    #
    SIC_interp = SIC_flat[idx["ice_charts_to_UNet"]]
    LSM_interp = LSM_flat[idx["ice_charts_to_UNet"]]
    #
    Data_UNet_grid["x"] = np.copy(UNet_coordinates["x"])
    Data_UNet_grid["y"] = np.copy(UNet_coordinates["y"])
    Data_UNet_grid["lat"] = np.copy(UNet_coordinates["lat"])
    Data_UNet_grid["lon"] = np.copy(UNet_coordinates["lon"])
    SIC_UNet_grid = np.reshape(SIC_interp, (len(UNet_coordinates["y"]), len(UNet_coordinates["x"])), order = "C")
    Data_UNet_grid["LSM"] = np.reshape(LSM_interp, (len(UNet_coordinates["y"]), len(UNet_coordinates["x"])), order = "C")
    #
    SIC_UNet_grid[Data_UNet_grid["LSM"] == 0] = np.nan
    Data_UNet_grid["SIC"] = np.copy(SIC_UNet_grid)
    #
    return(Data_UNet_grid)


# In[170]:


def write_netcdf(date_task, Data_UNet_grid, paths):
    path_output = paths["output"] + date_task[0:4] + "/" + date_task[4:6] + "/"
    if os.path.isdir(path_output) == False:
        os.system("mkdir -p " + path_output)
    #
    file_output = path_output + "Ice_charts_svalbard_UNet_grid_" + date_task + ".nc"
    output_netcdf = netCDF4.Dataset(file_output, "w", format = "NETCDF4")
    #
    x = output_netcdf.createDimension("x", len(Data_UNet_grid["x"]))
    y = output_netcdf.createDimension("y", len(Data_UNet_grid["y"]))
    #
    stereographic = output_netcdf.createVariable("crs", "int")
    x = output_netcdf.createVariable("x", "d", ("x"))
    y = output_netcdf.createVariable("y", "d", ("y"))
    lat = output_netcdf.createVariable("lat", "d", ("y","x"))
    lon = output_netcdf.createVariable("lon", "d", ("y","x"))
    SIC = output_netcdf.createVariable("SIC", "d", ("y","x"))
    LSM = output_netcdf.createVariable("LSM", "d", ("y","x"))
    #
    stereographic.grid_mapping_name = "polar_stereographic"
    stereographic.latitude_of_projection_origin = 90.0
    stereographic.longitude_of_projection_origin = -45.0
    stereographic.scale_factor_at_projection_origin = 1.0
    stereographic.straight_vertical_longitude_from_pole = -45.0
    stereographic.false_easting = 0.0
    stereographic.false_northing = 0.0
    stereographic.proj4_string = "+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere"
    x.standard_name = "projection_x_coordinate"
    x.units = "m"
    y.standard_name = "projection_y_coordinate"
    y.units = "m"
    lat.standard_name = "latitude"
    lat.units = "degrees_north"
    lon.standard_name = "longitude"
    lon.units = "degrees_east"
    SIC.standard_name = "sea_ice_concentration"
    SIC.units = "%"
    LSM.standard_name = "Land sea mask"
    LSM.unts = "1: ocean, 0: sea"
    #
    x[:] = Data_UNet_grid["x"]
    y[:] = Data_UNet_grid["y"]
    lat[:,:] = Data_UNet_grid["lat"]
    lon[:,:] = Data_UNet_grid["lon"]
    SIC[:,:] = Data_UNet_grid["SIC"]
    LSM[:,:] = Data_UNet_grid["LSM"]
    #
    output_netcdf.close()


# In[171]:


t0 = time.time()
UNet_coordinates = load_UNet_coordinates()
list_dates = make_list_dates(date_min, date_max)
for date_task in list_dates:
    filename_ice_charts = paths["ice_charts"] + date_task[0:4] + "/" + date_task[4:6] + "/" + "ice_conc_svalbard_" + date_task + "1500.nc"
    if os.path.isfile(filename_ice_charts):
        print(date_task)
        Data_ice_charts = load_ice_chart_data(filename_ice_charts)
        if "idx" in globals():
            pass
        else:
            idx = extract_idx(Data_ice_charts, UNet_coordinates, crs = crs)
        #
        Data_UNet_grid = nearest_neighbor_interp(idx, Data_ice_charts, UNet_coordinates)
        write_netcdf(date_task, Data_UNet_grid, paths)
tf = time.time()
print("Computing time", tf - t0)

