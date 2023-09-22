#$ -S /bin/bash
#$ -l h_rt=00:15:00
#$ -q research-el7.q
#$ -l h_vmem=6G
#$ -t 1-424
#$ -o /home/cyrilp/Documents/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /home/cyrilp/Documents/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -cwd

source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate production-10-2022

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

cat > "/home/cyrilp/Documents/PROG/Training_data_COSI_""$SGE_TASK_ID"".py" << EOF
###################################################################################################

#!/usr/bin/env python
# coding: utf-8

# In[166]:


import os 
import scipy
import netCDF4
import numpy as np
import pyproj
import datetime 
import sklearn.linear_model
import time


# Constants

# In[167]:

#
date_min = "20130103"
date_max = "20201231"
#
SIC_trend_period = 5
lead_time_max = 10
#
paths = {}
paths["output"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Training/Land_free_ocean/"
paths["LSM"] = "/lustre/storeB/project/copernicus/svalnav/Data/TOPAZ4/"
paths["TOPAZ"] = "/lustre/storeB/project/copernicus/ARC-MFC/ARC-METNO-ARC-TOPAZ4_2_PHYS-FOR/arctic/mersea-class1/"
paths["ECMWF"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/ECMWF/"
paths["AMSR2"] = "/lustre/storeB/project/copernicus/cosi/WP2/SIC/v0.1/"
# until 2020
paths["OSISAF"] = "/lustre/storeB/project/copernicus/osisaf/data/reprocessed/ice/conc/v3p0/"
# from 2021
#paths["OSISAF"] = "/lustre/storeB/project/copernicus/osisaf/data/reprocessed/ice/conc-cont-reproc/v3p0/"
#
proj = {}
proj["TOPAZ"] = "+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere"
proj["ECMWF"] = "+proj=latlon"
proj["OSISAF"] = "+proj=laea +lon_0=0 +datum=WGS84 +ellps=WGS84 +lat_0=90.0"
proj["AMSR2"] = "+ellps=WGS84 +lat_0=90 +lon_0=0 +no_defs=None +proj=laea +type=crs +units=m +x_0=0 +y_0=0"
#
crs = {}
for var in proj:
    crs[var] = pyproj.CRS.from_proj4(proj[var])
#
variables = {}
variables["LSM"] = ["LSM"]
variables["TOPAZ"] = ["fice", "hice", "u", "v"]
variables["ECMWF"] = ["U10M", "V10M", "T2M"]
variables["OSISAF"] = ["ice_conc", "trend"]  # trend is calculated from the last n days before the forecast start date
variables["AMSR2"] = ["ice_conc", "total_standard_uncertainty"] 


# task_date function
# 
#     date_min: earliest forecast start date to process
#     date_max: latest forecast start date to process
#     task_ID: task ID when parallelizing (SGE_TASK_ID)

# In[168]:


def task_date(date_min, date_max, task_ID):
    current_date = datetime.datetime.strptime(date_min, '%Y%m%d')
    end_date = datetime.datetime.strptime(date_max, '%Y%m%d')
    list_date = []
    while current_date <= end_date:
        list_date.append(current_date.strftime('%Y%m%d'))
        current_date = current_date + datetime.timedelta(days = 7)
    date_task = list_date[task_ID - 1]
    return(date_task)


# ECMWF_time_steps_to_daily_time_steps function => Compute the daily mean of the variable for each days  
#     
#     time_ECMWF: time variable in ECMWF netCDF files
#     field: 2D array variable
#     ndays: number of days (lead time) to compute 

# In[169]:


def ECMWF_time_steps_to_daily_time_steps(time_ECMWF, field, ndays):
    lead_time = time_ECMWF - time_ECMWF[0]
    ts_start = np.linspace(0 * 24, (ndays - 1) * 24, ndays)
    ts_end = ts_start + 24
    daily_field = np.full((ndays, field.shape[1], field.shape[2]), np.nan)
    #
    for ts in range(0, ndays):
        lead_time_idx = np.squeeze(np.where(np.logical_and(lead_time >= ts_start[ts], lead_time < ts_end[ts])))
        if ts == 3:
            daily_field[ts,:,:] = (18 * np.nanmean(np.ma.squeeze(field[lead_time_idx[0:18],:,:]), axis = 0) \
                                  + 6 * np.nanmean(np.ma.squeeze(field[lead_time_idx[18:20],:,:]), axis = 0)) / 24
        else:
            daily_field[ts,:,:] = np.nanmean(np.ma.squeeze(field[lead_time_idx,:,:]), axis = 0)
    #
    return(daily_field)


# Padding function (make_padding)  
# 
#     x and y must be vectors (can be latitude / longitude if the data are on a regular grid)  
#     field must be either a 2D array (y, x) or a 3D array (time, y, x)

# In[170]:


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


# Regridding functions (nearest_neighbor_indexes and nearest_neighbor_interp)  
# 
#     xx_input and yy_input must be 2D arrays
#     x_output and y_output must be vectors  
#     field must be either a 2D array with dimensions (y, x) or a 3D array with dimensions (time, y, x) 
#     invalid_values = fill value to replace by 0. Land is therefore considered as open ocean.

# In[171]:


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


# In[172]:


def nearest_neighbor_interp(xx_input, yy_input, x_output, y_output, field, fill_value = None):
    xx_input_flat = np.ndarray.flatten(xx_input)
    yy_input_flat = np.ndarray.flatten(yy_input)
    #
    if fill_value is not None:
        if field.ndim == 2:
            idx_fill_value = np.ndarray.flatten(field) == fill_value
        elif field.ndim == 3:
            idx_fill_value = np.ndarray.flatten(field[0,:,:]) == fill_value
        #
        xx_input_flat = xx_input_flat[idx_fill_value == False]
        yy_input_flat = yy_input_flat[idx_fill_value == False]
    #
    xx_output, yy_output = np.meshgrid(x_output, y_output)
    xx_output_flat = np.ndarray.flatten(xx_output)
    yy_output_flat = np.ndarray.flatten(yy_output)
    #
    idx = nearest_neighbor_indexes(xx_input_flat, yy_input_flat, xx_output_flat, yy_output_flat)
    #
    if field.ndim == 2:
        field_flat = np.ndarray.flatten(field)
        if fill_value is not None:
            field_flat = field_flat[idx_fill_value == False]
        #
        field_interp = field_flat[idx]
        field_regrid = np.reshape(field_interp, (len(y_output), len(x_output)), order = "C")
    #    
    elif field.ndim == 3:
        time_dim = len(field[:,0,0])
        field_regrid = np.full((time_dim, len(y_output), len(x_output)), np.nan)
        #
        for t in range(0, time_dim):
            field_flat = np.ndarray.flatten(field[t,:,:])
            if fill_value is not None:
                field_flat = field_flat[idx_fill_value == False]
            #
            field_interp = field_flat[idx]
            field_regrid[t,:,:] = np.reshape(field_interp, (len(y_output), len(x_output)), order = "C")
    #
    return(field_regrid)


# rotate_wind function
# 
#     x_wind, y_wind, lats, lons must be numpy arrays

# In[173]:


def rotate_wind(x_wind, y_wind, lats, lons, proj_str_from, proj_str_to):
    if np.shape(x_wind) != np.shape(y_wind):
        raise ValueError(f"x_wind {np.shape(x_wind)} and y_wind {np.shape(y_wind)} arrays must be the same size")
    if len(lats.shape) != 1:
        raise ValueError(f"lats {np.shape(lats)} must be 1D")
    if np.shape(lats) != np.shape(lons):
        raise ValueError(f"lats {np.shape(lats)} and lats {np.shape(lons)} must be the same size")
    if len(np.shape(x_wind)) == 1:
        if np.shape(x_wind) != np.shape(lats):
            raise ValueError(f"x_wind {len(x_wind)} and lats {len(lats)} arrays must be the same size")
    elif len(np.shape(x_wind)) == 2:
        if x_wind.shape[1] != len(lats):
            raise ValueError(f"Second dimension of x_wind {x_wind.shape[1]} must equal number of lats {len(lats)}")
    else:
        raise ValueError(f"x_wind {np.shape(x_wind)} must be 1D or 2D")
    #
    proj_from = pyproj.Proj(proj_str_from)
    proj_to = pyproj.Proj(proj_str_to)
    transformer = pyproj.transformer.Transformer.from_proj(proj_from, proj_to)
    #
    orig_speed = np.sqrt(x_wind**2 + y_wind**2)
    #
    x0, y0 = proj_from(lons, lats)
    if proj_from.name != "longlat":
        x1 = x0 + x_wind
        y1 = y0 + y_wind
    else:
        factor = 3600000.0
        x1 = x0 + x_wind / factor / np.cos(lats * 3.14159265 / 180)
        y1 = y0 + y_wind / factor
    #
    X0, Y0 = transformer.transform(x0, y0)
    X1, Y1 = transformer.transform(x1, y1)
    #
    new_x_wind = X1 - X0
    new_y_wind = Y1 - Y0
    #
    if proj_to.name == "longlat":
        new_x_wind *= np.cos(lats * 3.14159265 / 180)
    #
    if proj_to.name == "longlat" or proj_from.name == "longlat":
        curr_speed = np.sqrt(new_x_wind**2 + new_y_wind**2)
        new_x_wind *= orig_speed / curr_speed
        new_y_wind *= orig_speed / curr_speed
    #
    return(new_x_wind, new_y_wind)


# read_netCDF functions  
# 
#     filename: filename including the path
#     variables: list of variables (excluding time, x, y, lat, lon) to extract (list of strings)
#     paths: dictionary defined in the Constants section

# In[174]:


def read_netCDF(filename, variables, paths = paths):
    Dataset = {}
    nc = netCDF4.Dataset(filename, "r")
    #
    if (paths["LSM"] in filename) == False:
        Dataset["time"] = nc.variables["time"][:]
    #
    if (paths["LSM"] in filename) or (paths["TOPAZ"] in filename):
        xmin = 65
        xmax = 609
        ymin = 70
        ymax = 614
        Dataset["x"] = nc.variables["x"][xmin:xmax] * 100 * 1000
        Dataset["y"] = nc.variables["y"][ymin:ymax] * 100 * 1000
        Dataset["lat"] = nc.variables["latitude"][ymin:ymax, xmin:xmax] 
        Dataset["lon"] = nc.variables["longitude"][ymin:ymax, xmin:xmax]
    #
    elif paths["ECMWF"] in filename:
        Dataset["lat"] = nc.variables["lat"][:]
        Dataset["lon"] = nc.variables["lon"][:]
    #
    elif (paths["OSISAF"] in filename) or (paths["AMSR2"] in filename):
        Dataset["x"] = nc.variables["xc"][:] * 1000
        Dataset["y"] = nc.variables["yc"][:] * 1000
        Dataset["lat"] = nc.variables["lat"][:,:] 
        Dataset["lon"] = nc.variables["lon"][:,:]
    #
    for var in variables:
        vardim = nc.variables[var].ndim
        if vardim == 1:
            Dataset[var] = nc.variables[var][:]
        elif vardim == 2:
            if (paths["LSM"] in filename) or (paths["TOPAZ"] in filename):
                Dataset[var] = nc.variables[var][ymin:ymax, xmin:xmax]
            else:
                Dataset[var] = nc.variables[var][:,:]
        elif vardim == 3:
            if paths["TOPAZ"] in filename:
                if var == "fice":
                    Dataset[var] = nc.variables[var][:, ymin:ymax, xmin:xmax] * 100
                else:
                    Dataset[var] = nc.variables[var][:, ymin:ymax, xmin:xmax]
            else:
                Dataset[var] = nc.variables[var][:,:,:]
        elif vardim == 4:
            if paths["TOPAZ"] in filename:
                Dataset[var] = nc.variables[var][:,0,ymin:ymax, xmin:xmax]
        else:
            print("ERROR. Number of dimensions higher than 4.")
    nc.close()
    #
    return(Dataset)


# extract_TOPAZ_data function
# 
#     date_task: forecast start date of the TOPAZ forecasts
#     ndays: maximum lead time in days
#     variables: list of variables to extract (variables["TOPAZ"]) 
#     paths = path where the TOPAZ data are stored (paths["TOPAZ"]). Defined in the Constant section

# In[175]:


def extract_TOPAZ_data(date_task, ndays, variables = variables["TOPAZ"], paths = paths):
    dataset = []
    for lt in range(0, ndays):
        forecast_datetime = datetime.datetime.strptime(date_task, "%Y%m%d") + datetime.timedelta(days = lt)
        forecast_date = forecast_datetime.strftime("%Y%m%d")
        filename_TOPAZ = paths["TOPAZ"] + forecast_date + "_dm-metno-MODEL-topaz4-ARC-b" + date_task + "-fv02.0.nc"
        dataset.append(filename_TOPAZ)
    #
    if len(dataset) == ndays:
        #
        Data_TOPAZ = {}
        for lt in range(0, ndays):
            Data_lt = read_netCDF(dataset[lt], variables)
            if lt == 0:
                Data_TOPAZ = Data_lt.copy()
            else:
                Data_TOPAZ["time"] = np.concatenate((Data_TOPAZ["time"], Data_lt["time"]), axis = 0)
                for var in variables:
                    Data_TOPAZ[var] = np.concatenate((Data_TOPAZ[var], Data_lt[var]), axis = 0)
        #
        Data_output = {}
        Data_output["time"] = np.copy(Data_TOPAZ["time"])
        Data_output["x"] = np.copy(Data_TOPAZ["x"])
        Data_output["y"] = np.copy(Data_TOPAZ["y"])
        Data_output["lat"] = np.copy(Data_TOPAZ["lat"])
        Data_output["lon"] = np.copy(Data_TOPAZ["lon"])
        #
        for var in variables:
            if (var == "fice") or (var == "hice"):
                idx_assign = np.logical_or(np.isnan(Data_TOPAZ[var]) == True, Data_TOPAZ[var] == -32767)
                Data_output[var] = np.copy(Data_TOPAZ[var])
                Data_output[var][idx_assign == True] =  0
            elif (var == "u") or (var == "v"):
                Data_output[var + "_cum"] = np.full(np.shape(Data_TOPAZ[var]), np.nan)
                for lt in range(0, ndays):
                    Data_output[var + "_cum"][lt,:,:] = np.nanmean(Data_TOPAZ[var][0:lt+1,:,:], axis = 0)
                #
                idx_assign = np.logical_or(np.isnan(Data_output[var + "_cum"]) == True, Data_output[var + "_cum"] == -32767)
                Data_output[var + "_cum"][idx_assign == True] =  0
    else:
        print("ERROR. Some forecast dates are missing in the TOPAZ dataset")
    #
    return(Data_output)


# extract_ECMWF_data function   
# 
#     filename: filename (including path) containing ECMWF data 
#     ndays: maximum lead time in days
#     TOPAZ: TOPAZ dataset (dictionary)   
#     proj: dictionary of proj4 strings
#     variables: list of variables to extract (variables["ECMWF"])

# In[176]:


def extract_ECMWF_data(filename, ndays, TOPAZ, proj = proj, variables = variables["ECMWF"], crs = crs):
    ECMWF = read_netCDF(filename, variables)
    Data_TOPAZgrid = {}
    transform_ECMWF_to_TOPAZ = pyproj.Transformer.from_crs(crs["ECMWF"], crs["TOPAZ"], always_xy = True)
    lons, lats = np.meshgrid(ECMWF["lon"], ECMWF["lat"])
    xx_ECMWF_TOPAZproj, yy_ECMWF_TOPAZproj = transform_ECMWF_to_TOPAZ.transform(lons, lats)
    #
    Data_ECMWFgrid = {}
    for var in variables:
        Data_ECMWFgrid[var] = ECMWF_time_steps_to_daily_time_steps(ECMWF["time"], ECMWF[var], ndays)
    #
    if ("U10M" in Data_ECMWFgrid) and ("V10M" in Data_ECMWFgrid):
        x_wind = np.full((len(TOPAZ["time"]), len(ECMWF["lat"]), len(ECMWF["lon"])), np.nan)
        y_wind = np.full((len(TOPAZ["time"]), len(ECMWF["lat"]), len(ECMWF["lon"])), np.nan)
        #
        for ts in range(0, len(TOPAZ["time"])):
            x_wind_rot, y_wind_rot = rotate_wind(np.ndarray.flatten(Data_ECMWFgrid["U10M"][ts,:,:]), 
                                                 np.ndarray.flatten(Data_ECMWFgrid["V10M"][ts,:,:]),
                                                 np.ndarray.flatten(lats), 
                                                 np.ndarray.flatten(lons), 
                                                 proj["ECMWF"], 
                                                 proj["TOPAZ"]
                                                )
            #
            x_wind[ts,:,:] = np.reshape(x_wind_rot, (len(ECMWF["lat"]), len(ECMWF["lon"])), order = "C")
            y_wind[ts,:,:] = np.reshape(y_wind_rot, (len(ECMWF["lat"]), len(ECMWF["lon"])), order = "C")            
            #
        Data_ECMWFgrid["wind_x"] = np.copy(x_wind)
        Data_ECMWFgrid["wind_y"] = np.copy(y_wind)
        Data_ECMWFgrid.pop("U10M")
        Data_ECMWFgrid.pop("V10M")
    #
    Cum_data_ECMWFgrid = {}
    for var in Data_ECMWFgrid:
        var_cum = np.full((len(TOPAZ["time"]), len(ECMWF["lat"]), len(ECMWF["lon"])), np.nan)
        for ts in range(0, len(TOPAZ["time"])):
            var_cum[ts,:,:] = np.nanmean(Data_ECMWFgrid[var][0:ts+1,:,:], axis = 0)
        #
        Cum_data_ECMWFgrid[var + "_cum"] = np.copy(var_cum)
        Cum_data_ECMWFgrid[var + "_cum"][np.isnan(var_cum) == True] = -32767
    #
    for var in Cum_data_ECMWFgrid:
        Data_TOPAZgrid[var] = nearest_neighbor_interp(xx_ECMWF_TOPAZproj, yy_ECMWF_TOPAZproj, TOPAZ["x"], TOPAZ["y"], Cum_data_ECMWFgrid[var], fill_value = -32767)
    #
    return(Data_TOPAZgrid)


# SIC_trend function
# 
#     date_task: forecast start date (string "YYYYMMDD")
#     trend_period: Number of days to take into account for calculating the trend

# In[177]:


def SIC_trend(date_task, trend_period):
    Dataset = {}
    first_date = datetime.datetime.strptime(date_task, "%Y%m%d") - datetime.timedelta(days = trend_period)
    last_date = datetime.datetime.strptime(date_task, "%Y%m%d") - datetime.timedelta(days = 1)
    current_date = first_date
    time_vect = []
    #
    for id in range(0, trend_period):
        date_str = current_date.strftime("%Y%m%d")
        if int(date_task[0:4]) < 2021:
            filename_SIC = paths["OSISAF"] + date_str[0:4] + "/" + date_str[4:6] + "/" + "ice_conc_nh_ease2-250_cdr-v3p0_" + date_str + "1200.nc"
        else:
            filename_SIC = paths["OSISAF"] + date_str[0:4] + "/" + date_str[4:6] + "/" + "ice_conc_nh_ease2-250_icdr-v3p0_" + date_str + "1200.nc"
        #
        if os.path.isfile(filename_SIC):
            time_vect.append(id)
            SIC_data = read_netCDF(filename_SIC, variables = ["ice_conc"])
            #
            if "SIC_all" in locals():
                SIC_all = np.concatenate((SIC_all, SIC_data["ice_conc"]), axis = 0)
            else:
                SIC_all = np.copy(SIC_data["ice_conc"])
            #
            if current_date == last_date:
                for var in SIC_data:
                    Dataset[var] = np.copy(SIC_data[var])
            #
        current_date = current_date + datetime.timedelta(days = 1)
    #
    if len(time_vect) >= 3:
        trend = np.zeros((SIC_all.shape[1], SIC_all.shape[2]))
        x = np.array(time_vect).reshape((-1,1))
        for i in range(0, SIC_all.shape[1]):
            for j in range(0, SIC_all.shape[2]):
                y = SIC_all[:,i,j]
                if np.all(y >= 0):
                    model = sklearn.linear_model.LinearRegression().fit(x, y)
                    trend[i,j] = model.coef_[0]
        Dataset["trend"] = np.copy(trend)
    #
    return(Dataset)


# extract_SIC_obs_predictors function
#     
#     date_task: forecast start date (string "YYYYMMDD")
#     trend_period: Number of days to take into account for calculating the trend
#     TOPAZ: TOPAZ dataset (dictionary)   
#     LSM: TOPAZ land sea mask (dictionary)
#     proj: dictionary of proj4 strings
#     variables: list of variables to extract (variables["OSISAF"])
#     crs: crs defined in "Constants"

# In[178]:


def extract_SIC_obs_predictors(date_task, trend_period, TOPAZ, LSM, proj = proj, variables = variables["OSISAF"], crs = crs):
    Data_TOPAZgrid = {}
    SIC_obs = SIC_trend(date_task, trend_period)
    #
    transform_SIC_to_TOPAZ = pyproj.Transformer.from_crs(crs["OSISAF"], crs["TOPAZ"], always_xy = True)
    xx_SIC_obs, yy_SIC_obs = np.meshgrid(SIC_obs["x"], SIC_obs["y"])
    xx_SIC_TOPAZproj, yy_SIC_TOPAZproj = transform_SIC_to_TOPAZ.transform(xx_SIC_obs, yy_SIC_obs)
    #
    for var in variables:
        SIC_obs[var] = np.squeeze(SIC_obs[var].data)
        Data_TOPAZgrid[var] = nearest_neighbor_interp(xx_SIC_TOPAZproj, yy_SIC_TOPAZproj, TOPAZ["x"], TOPAZ["y"], SIC_obs[var], fill_value = -32767)
        Data_TOPAZgrid[var][LSM["LSM"] == 0] = 0
    #
    return(Data_TOPAZgrid)


# extract_targets (MUST BE UPDATED WITH AMSR2 SPECIFICATIONS)
# 
#     date_task: forecast start date (string "YYYYMMDD")
#     ndays: Number of days to take into account for calculating the trend
#     TOPAZ: TOPAZ dataset (dictionary)
#     LSM: TOPAZ land sea maskLSM: TOPAZ land sea mask
#     proj: dictionary of proj4 strings
#     variables: list of variables to extract (variables["OSISAF"])
#     crs: crs defined in "Constants"

# In[179]:


def extract_targets(date_task, ndays, TOPAZ, LSM, proj = proj, crs = crs, paths = paths, variables = variables):
    Data_TOPAZgrid = {}
    #
    transform_SIC_to_TOPAZ = pyproj.Transformer.from_crs(crs["AMSR2"], crs["TOPAZ"], always_xy = True)
    #
    for lt in range(0, ndays):
        date_str = (datetime.datetime.strptime(date_task, "%Y%m%d") + datetime.timedelta(days = lt)).strftime("%Y%m%d%H%M")
        date_day_after_str = (datetime.datetime.strptime(date_task, "%Y%m%d") + datetime.timedelta(days = lt + 1)).strftime("%Y%m%d%H%M")
        filename_SIC = paths["AMSR2"] + date_str[0:4] + "/" + date_str[4:6] + "/" + "sic_cosi-5km_" + date_str + "-" + date_day_after_str + ".nc"
        SIC_obs = read_netCDF(filename_SIC, variables = variables["AMSR2"])
        #
        if lt == 0:
            xx_SIC_obs, yy_SIC_obs = np.meshgrid(SIC_obs["x"], SIC_obs["y"])
            xx_SIC_TOPAZproj, yy_SIC_TOPAZproj = transform_SIC_to_TOPAZ.transform(xx_SIC_obs, yy_SIC_obs)
        #
        SIC_TOPAZgrid = nearest_neighbor_interp(xx_SIC_TOPAZproj, yy_SIC_TOPAZproj, TOPAZ["x"], TOPAZ["y"], SIC_obs["ice_conc"], fill_value = -32767)
        total_uncertainty_TOPAZgrid = nearest_neighbor_interp(xx_SIC_TOPAZproj, yy_SIC_TOPAZproj, TOPAZ["x"], TOPAZ["y"], SIC_obs["total_standard_uncertainty"], fill_value = -32767)
        #
        ls_mask = np.expand_dims(LSM["LSM"], axis = 0)
        SIC_TOPAZgrid[ls_mask == 0] = 0
        total_uncertainty_TOPAZgrid[ls_mask == 0] = 0
        #
        if lt == 0:
            Data_TOPAZgrid["SIC"] = np.copy(SIC_TOPAZgrid)
            Data_TOPAZgrid["SIC_total_standard_uncertainty"] = np.copy(total_uncertainty_TOPAZgrid)
        else:
            Data_TOPAZgrid["SIC"] = np.concatenate((Data_TOPAZgrid["SIC"], SIC_TOPAZgrid), axis = 0)
            Data_TOPAZgrid["SIC_total_standard_uncertainty"] = np.concatenate((Data_TOPAZgrid["SIC_total_standard_uncertainty"], total_uncertainty_TOPAZgrid), axis = 0)
    #
    Data_TOPAZgrid["SIE_10"] = np.zeros(np.shape(Data_TOPAZgrid["SIC"]))
    Data_TOPAZgrid["SIE_20"] = np.zeros(np.shape(Data_TOPAZgrid["SIC"]))
    Data_TOPAZgrid["SIE_10"][Data_TOPAZgrid["SIC"] >= 10] = 1
    Data_TOPAZgrid["SIE_20"][Data_TOPAZgrid["SIC"] >= 20] = 1
    #
    Data_TOPAZgrid["TOPAZ_error"] = TOPAZ["fice"] - Data_TOPAZgrid["SIC"]
    #
    return(Data_TOPAZgrid)


# write_netCDF function
# 
#     date_task: forecast start date (string "YYYYMMDD")
#     Datasets: Dictionary containing all variables that we want to extract
#     paths: paths defined in the Constants section
#     trend_period: Number of days to take into account for calculating the trend

# In[180]:


def write_netCDF(date_task, Datasets, paths, trend_period):
    Outputs = vars()
    #
    path_output = paths["output"] + date_task[0:4] + "/" + date_task[4:6] + "/"
    if os.path.exists(path_output) == False:
        os.system("mkdir -p " + path_output)    
    output_filename = path_output + "Dataset_" + date_task + ".nc"
    if os.path.isfile(output_filename):
        os.system("rm " + output_filename)
    output_netcdf = netCDF4.Dataset(output_filename, 'w', format = 'NETCDF4')
    #
    dimensions = ["time", "x", "y"]
    for di in dimensions:
        Outputs[di] = output_netcdf.createDimension(di, len(Datasets["TOPAZ"][di]))
    #
    dim_variables = dimensions + ["lat", "lon"]
    for dv in dim_variables:
        if Datasets["TOPAZ"][dv].ndim == 1:
            Outputs[dv] = output_netcdf.createVariable(dv, "d", (dv))
            Outputs[dv][:] = Datasets["TOPAZ"][dv]
            if dv == "time":
                Outputs[dv].standard_name = "forecast time"
                Outputs[dv].units = "hours since 1950-1-1T00:00:00Z"
            elif dv == "x" or dv == "y":
                Outputs[dv].standard_name = "projection_" + dv + "_coordinate"
                Outputs[dv].units = "m"
        elif Datasets["TOPAZ"][dv].ndim == 2:
            Outputs[dv] = output_netcdf.createVariable(dv, "d", ("y", "x"))
            Outputs[dv][:,:] = Datasets["TOPAZ"][dv]
            if dv == "lat":
                Outputs[dv].standard_name = "latitude"
            elif dv == "lon":
                Outputs[dv].standard_name = "longitude"
            Outputs[dv].units = "degrees"
    #
    SIC_variables = ["ice_conc", "fice", "SIC"]
    for ds in Datasets:
        for var in Datasets[ds]:
            if (var in dim_variables) == False:
                if var == "LSM":
                    var_name = "LSM"
                elif var in SIC_variables:
                    var_name = ds + "_SIC"
                else:
                    var_name = ds + "_" + var
                #
                if Datasets[ds][var].ndim == 2:
                    Outputs[var_name] = output_netcdf.createVariable(var_name, "d", ("y", "x"))
                    Outputs[var_name][:,:] = np.round(Datasets[ds][var], 3)
                elif Datasets[ds][var].ndim == 3:
                    Outputs[var_name] = output_netcdf.createVariable(var_name, "d", ("time", "y", "x"))
                    Outputs[var_name][:,:,:] = np.round(Datasets[ds][var], 3)
                #
                if var in SIC_variables:
                    if ds == "TARGET":
                        Outputs[var_name].standard_name = "AMSR2 sea ice concentration"
                    else: 
                        Outputs[var_name].standard_name = ds + " sea ice concentration"
                    Outputs[var_name].units = "%"
                if "SIE_" in var:
                    Outputs[var_name].standard_name = "Sea ice extent with a sea ice concentration higher than " + var[-2:len(var)] + " %"
                    Outputs[var_name].units = "1 if sea ice concentration higher than " + var[-2:len(var)] + " %, 0 otherwise"
                elif var == "SIC_total_standard_uncertainty":
                    Outputs[var_name].standard_name = "Total uncertainty (one standard deviation) of concentration of sea ice"
                    Outputs[var_name].units = "%"
                elif var == "trend":
                    Outputs[var_name].standard_name = "Sea ice concentration trend over the " + str(SIC_trend_period) + " days preceding the forecast start date"
                    Outputs[var_name].units = "% / day"
                elif var == "TOPAZ_error":
                    Outputs[var_name].standard_name = "TOPAZ error in sea ice concentration (TOPAZ - AMSR2 observations)"
                    Outputs[var_name].units = "%"
                elif ds + "_" + var == "TOPAZ_hice":
                    Outputs[var_name].standard_name = "TOPAZ sea ice thickness"
                    Outputs[var_name].units = "m"
                elif var == "u_cum":
                    Outputs[var_name].standard_name = "Mean x sea water velocity since the forecast start date"
                    Outputs[var_name].units = "m.s-1"
                elif var == "v_cum":
                    Outputs[var_name].standard_name = "Mean y sea water velocity since the forecast start date"
                    Outputs[var_name].units = "m.s-1"
                elif var == "LSM":
                    Outputs[var_name].standard_name = "Land sea mask"
                    Outputs[var_name].units = "1: ocean, 0: land"
                elif ds + "_" + var == "ECMWF_T2M_cum":
                    Outputs[var_name].standard_name = "Mean ECMWF 2 metre temperature since the forecast start date"
                    Outputs[var_name].units = "K"
                elif ds + "_" + var == "ECMWF_wind_x_cum":
                    Outputs[var_name].standard_name = "Mean ECMWF wind in the x direction since the forecast start date"
                    Outputs[var_name].units = "m/s"
                elif ds + "_" + var == "ECMWF_wind_y_cum":
                    Outputs[var_name].standard_name = "Mean ECMWF wind in the y direction since the forecast start date"
                    Outputs[var_name].units = "m/s"
    output_netcdf.close()    


# Data processing

# In[181]:


t0 = time.time()
#
date_task = task_date(date_min, date_max, task_ID = $SGE_TASK_ID)
print("date_task", date_task)
#
filename_LSM = paths["LSM"] + "TOPAZ4_land_sea_mask.nc"            #
filename_ECMWF = paths["ECMWF"] + date_task[0:4] + "/" + date_task[4:6] + "/ECMWF_operational_forecasts_T2m_10mwind_" + date_task + "_NH.nc"
#
Datasets = {}
#
Datasets["LSM"] = read_netCDF(filename = filename_LSM, 
                              variables = variables["LSM"])   # 1 ocean / 0 land
#
Datasets["TOPAZ"] = extract_TOPAZ_data(date_task = date_task, 
                                       ndays = lead_time_max, 
                                       variables = variables["TOPAZ"], 
                                       paths = paths)
#
Datasets["ECMWF"] = extract_ECMWF_data(filename = filename_ECMWF, 
                                       ndays = lead_time_max, 
                                       TOPAZ = Datasets["TOPAZ"], 
                                       variables = variables["ECMWF"], 
                                       crs = crs)
#
Datasets["SICobs"] = extract_SIC_obs_predictors(date_task = date_task, 
                                                trend_period = SIC_trend_period, 
                                                TOPAZ = Datasets["TOPAZ"], 
                                                LSM = Datasets["LSM"], 
                                                proj = proj, variables = variables["OSISAF"], 
                                                crs = crs)
#
Datasets["TARGET"] = extract_targets(date_task = date_task, 
                                     ndays = lead_time_max, 
                                     TOPAZ = Datasets["TOPAZ"], 
                                     LSM = Datasets["LSM"], 
                                     proj = proj, 
                                     crs = crs, 
                                     paths = paths)
#
write_netCDF(date_task = date_task, 
             Datasets = Datasets, 
             paths = paths, 
             trend_period = SIC_trend_period)
#
t1 = time.time()


# In[182]:


dt_all = t1 - t0
#
print("dt_all", dt_all)
###################################################################################################
EOF
python3 "/home/cyrilp/Documents/PROG/Training_data_COSI_""$SGE_TASK_ID"".py"


