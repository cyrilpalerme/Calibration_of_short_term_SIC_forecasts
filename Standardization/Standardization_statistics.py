#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import h5py
import netCDF4
import datetime
import numpy as np
import time


# Constants

# In[2]:


date_min = "20130103"
date_max = "20201231"
#
paths = {}
paths["data"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Training_HRES/Land_free_ocean/"
paths["output"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Training_HRES/Standardization/"
#
list_variables = {}
list_variables["data"] = []
list_variables["geolocation"] = ["time", "x", "y", "lat", "lon"]
#
frequency = "weekly"


# extract_dataset function
# 
#     date_min: earliest date to consider
#     date_max: latest date to consider
#     frequency: frequency of the forecasts (weekly or daily)
#     paths: paths from the "Constants" section

# In[3]:


def extract_dataset(date_min, date_max, frequency, paths = paths):
    current_date = datetime.datetime.strptime(date_min, '%Y%m%d')
    end_date = datetime.datetime.strptime(date_max, '%Y%m%d')
    dataset = []
    while current_date <= end_date:
        cdate = current_date.strftime('%Y%m%d')
        filename = paths["data"] + cdate[0:4] + "/" + cdate[4:6] + "/" + "Dataset_" + cdate + ".nc"
        if os.path.isfile(filename):
            dataset.append(filename)
        #
        if frequency == "daily":
            current_date = current_date + datetime.timedelta(days = 1)
        elif frequency == "weekly":
            current_date = current_date + datetime.timedelta(days = 7)
    #
    return(dataset)


# extract_variables function
# 
#     dataset: dataset created using the function "extract dataset"
#     list_variables: list_variables from the "Constants" section

# In[4]:


def extract_variables(dataset, list_variables = list_variables):
    nc = netCDF4.Dataset(dataset[0], "r")
    for var in nc.variables:
        if (var in list_variables["geolocation"]) == False:
            list_variables["data"].append(var)
    #
    list_variables["data"].append("initial_bias")
    list_variables["data"].append("TOPAZ_bias_corrected")
    #
    nc.close()
    return(list_variables)


# extract_stats function
#    
#     dataset: dataset created using the function "extract dataset"
#     variable_name: name of the variable which is going to be analyzed

# In[5]:


def extract_stats(dataset, variable_name):
    Stats = {}
    #
    if variable_name == "LSM":
        nc = netCDF4.Dataset(dataset[0], "r")
        field_conc = nc.variables["LSM"][:,:]
        nc.close()
        #
        Stats["min"] = np.nanmin(field_conc)
        Stats["max"] = np.nanmax(field_conc)
        Stats["std"] = np.nanstd(field_conc)
        Stats["mean"] = np.nanmean(field_conc)
    #
    elif (variable_name == "initial_bias") or (variable_name == "TOPAZ_bias_corrected"):
        for i, fi in enumerate(dataset):
            nc = netCDF4.Dataset(fi, "r")
            TOPAZ_SIC = nc.variables["TOPAZ_SIC"][:,:,:]
            SICobs_AMSR2 = nc.variables["SICobs_AMSR2_SIC"][:,:]
            ini_bias = np.expand_dims(TOPAZ_SIC[0,:,:] - SICobs_AMSR2, axis = 0)
            #
            if variable_name == "initial_bias":
                field = np.copy(ini_bias)
            elif variable_name == "TOPAZ_bias_corrected":
                ini_bias_3D = np.repeat(ini_bias, 10, axis = 0)
                field = np.expand_dims(TOPAZ_SIC - ini_bias_3D, axis = 0)
                field[field < 0] = 0
                field[field > 100] = 100
            #
            if i == 0:
                field_conc = np.copy(field)
            else:
                field_conc = np.concatenate((field_conc, field), axis = 0)
            nc.close()
        #
        Stats["min"] = np.nanmin(field_conc)
        Stats["max"] = np.nanmax(field_conc)
        Stats["std"] = np.nanstd(field_conc)
        Stats["mean"] = np.nanmean(field_conc)
    #
    elif ("ECMWF" in variable_name) or (variable_name == "TOPAZ_u_cum") or (variable_name == "TOPAZ_v_cum") or (variable_name == "TOPAZ_temperature_cum"):
        for i, fi in enumerate(dataset):
            nc = netCDF4.Dataset(fi, "r")
            field = np.expand_dims(nc.variables[variable_name][:,:,:], axis = 0)
            #
            if i == 0:
                field_conc = np.copy(field)
            else:
                field_conc = np.concatenate((field_conc, field), axis = 0)
            nc.close()
            #
        for lt in range(0, len(field[0,:,0,0])):
            if lt == 0:
                Stats["min"] = np.nanmin(field_conc[:,lt,:,:]) 
                Stats["max"] = np.nanmax(field_conc[:,lt,:,:])
                Stats["std"] = np.nanstd(field_conc[:,lt,:,:])
                Stats["mean"] = np.nanmean(field_conc[:,lt,:,:])
            else:
                Stats["min"] = np.hstack((Stats["min"], np.nanmin(field_conc[:,lt,:,:])))
                Stats["max"] = np.hstack((Stats["max"], np.nanmax(field_conc[:,lt,:,:])))
                Stats["std"] = np.hstack((Stats["std"], np.nanstd(field_conc[:,lt,:,:])))
                Stats["mean"] = np.hstack((Stats["mean"], np.nanmean(field_conc[:,lt,:,:])))
    #
    else:
        for i, fi in enumerate(dataset):
            nc = netCDF4.Dataset(fi, "r")
            if i == 0:
                vardim = nc.variables[variable_name].ndim
                LSM = nc.variables["LSM"][:,:]
                if vardim == 3:
                    LSM = np.expand_dims(LSM, axis = 0)
                    LSM = np.repeat(LSM, 10, axis = 0)
                LSM = np.ndarray.flatten(LSM)
            #
            if vardim == 2:
                field_flat = np.ndarray.flatten(nc.variables[variable_name][:,:])
            elif vardim == 3:
                field_flat = np.ndarray.flatten(nc.variables[variable_name][:,:,:])
            #
            field_flat = field_flat[LSM == 1]
            #   
            if i == 0:
                field_conc = np.copy(field_flat)
            else:
                field_conc = np.hstack((field_conc, field_flat))
            #
            nc.close()
        #
        Stats["min"] = np.nanmin(field_conc)
        Stats["max"] = np.nanmax(field_conc)
        Stats["std"] = np.nanstd(field_conc)
        Stats["mean"] = np.nanmean(field_conc)
    #
    return(Stats)


# write_hdf5 function
# 
#     Stats: output of the function "extract_stats"
#     date_min: date_min from the "Constants" section
#     date_max: date_max from the "Constants" section
#     frequency: frequency from the "Constants" section
#     paths: paths from the "Constants" section

# In[6]:


def write_hdf5(Stats, date_min = date_min, date_max = date_max, frequency = frequency, paths = paths):
    filename = paths["output"] + "Stats_standardization_" + date_min + "_" + date_max + "_" + frequency + ".h5"
    hf = h5py.File(filename, 'w')
    for var in Stats:
        for st in Stats[var]:
            output_var = var + "_" + st
            hf.create_dataset(output_var, data = Stats[var][st])
    hf.close()


# Data processing

# In[ ]:


t0 = time.time()
dataset = extract_dataset(date_min, date_max, frequency = frequency, paths = paths)
print("len(dataset)", len(dataset))
list_variables =  extract_variables(dataset, list_variables = list_variables)
#
Stats = {}
for var in list_variables["data"]:
    print(var)
    Stats[var] = extract_stats(dataset = dataset, variable_name = var)
#
write_hdf5(Stats, date_min = date_min, date_max = date_max, paths = paths)
#
t1 = time.time()

