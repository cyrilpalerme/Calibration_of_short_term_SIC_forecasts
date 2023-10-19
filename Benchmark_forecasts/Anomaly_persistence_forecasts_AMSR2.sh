#!/bin/bash -f
#$ -N SAP_COSI
#$ -l h_rt=00:00:30
#$ -S /bin/bash
#$ -pe shmem-1 1
#$ -l h_rss=1G,mem_free=1G,h_data=1G
#$ -q research-r8.q
#$ -t 1-1
##$ -j y
##$ -m ba
#$ -o /home/cyrilp/Documents/OUT/OUT_$JOB_NAME.$JOB_ID_$TASK_ID
#$ -e /home/cyrilp/Documents/ERR/ERR_$JOB_NAME.$JOB_ID_$TASK_ID
##$ -R y
##$ -r y

source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate production-10-2022

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

cat > "/home/cyrilp/Documents/PROG/Anomaly_forecasts_""$SGE_TASK_ID"".py" << EOF
###################################################################################################
#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import netCDF4
import datetime
import numpy as np


# In[8]:


#
date_min = "20201231"
date_max = "20221231"
#
clim_first_year = "2013"
clim_last_year = "2021"
#
lead_times = np.arange(11)
#
paths = {}
paths["training"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Training/Land_free_ocean/"
paths["AMSR2"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/AMSR2_TOPAZ4_grid/"
paths["clim"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Climatology/AMSR2/"
paths["output"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Anomaly_persistence_forecasts/AMSR2/"


# In[9]:


def make_list_dates(date_min, date_max):
    current_date = datetime.datetime.strptime(date_min, "%Y%m%d")
    end_date = datetime.datetime.strptime(date_max, "%Y%m%d")
    list_dates = []
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        list_dates.append(date_str)
        current_date = current_date + datetime.timedelta(days = 1)
    return(list_dates)


# In[10]:


def load_LSM(date_task = "20210101", paths = paths):
    filename = paths["training"] + date_task[0:4] + "/" + date_task[4:6] + "/" + "Dataset_" + date_task + ".nc"
    nc = netCDF4.Dataset(filename, "r")
    LSM = nc.variables["LSM"][:,:]
    nc.close()
    return(LSM)


# In[11]:


class Anomaly_persistence_forecasts():
    def __init__(self, forecast_start_date, lead_times, LSM, clim_first_year, clim_last_year, paths):
        self.paths = paths
        self.lead_times = lead_times
        self.forecast_start_date = forecast_start_date
        self.LSM = LSM
        self.clim_first_year = clim_first_year
        self.clim_last_year = clim_last_year
        self.anomaly_weights = np.linspace(0, 1, 21)
        self.start_date_doy = int(datetime.datetime.strptime(forecast_start_date, "%Y%m%d").strftime("%j"))
        if self.start_date_doy == 366:
            self.start_date_doy = 365
    #
    def make_anomaly_forecasts(self):
        filename_clim = self.paths["clim"] + "Climatology_SIC_" + self.clim_first_year + "_" + self.clim_last_year + ".nc"
        filename_SICobs = self.paths["AMSR2"] + self.forecast_start_date[0:4] + "/" + self.forecast_start_date[4:6] + "/" + "SIC_COSI_UNetgrid_" + self.forecast_start_date + ".nc"
        #
        nc_obs = netCDF4.Dataset(filename_SICobs, "r")
        SIC_obs = nc_obs.variables["SIC"][:,:]
        nc_obs.close()
        #
        nc_clim = netCDF4.Dataset(filename_clim, "r")
        SIC_clim = nc_clim.variables["SIC"][self.start_date_doy - 1,:,:]    
        nc_clim.close()
        #
        SIC_initial_anomaly = SIC_obs - SIC_clim
        SIC_anomaly_forecasts = np.full((len(lead_times), np.shape(SIC_initial_anomaly)[0], np.shape(SIC_initial_anomaly)[1]), np.nan)  
        #
        for lt, leadtime in enumerate(self.lead_times):
            target_date = (datetime.datetime.strptime(self.forecast_start_date, "%Y%m%d") + datetime.timedelta(days = int(leadtime))).strftime("%Y%m%d")
            target_doy = int(datetime.datetime.strptime(target_date, "%Y%m%d").strftime("%j"))
            if target_doy == 366:
                target_doy = 365
            #
            nc_clim_target = netCDF4.Dataset(filename_clim, "r")
            SIC_clim_target = nc_clim_target.variables["SIC"][target_doy - 1,:,:]
            nc_clim_target.close()
            SIC_anomaly_forecasts[lt,:,:] = SIC_initial_anomaly + SIC_clim_target
        #
        SIC_anomaly_forecasts[SIC_anomaly_forecasts < 0] = 0
        SIC_anomaly_forecasts[SIC_anomaly_forecasts > 100] = 100
        #
        return(SIC_anomaly_forecasts)
    #
    def write_netCDF(self, SIC_anomaly_forecasts):
        path_obs = self.paths["AMSR2"] + self.forecast_start_date[0:4] + "/" + self.forecast_start_date[4:6] + "/"
        filename_obs = path_obs + "SIC_COSI_UNetgrid_" + self.forecast_start_date + ".nc" 
        nc_obs = netCDF4.Dataset(filename_obs, "r")
        xc = nc_obs.variables["xc"][:]
        yc = nc_obs.variables["yc"][:]
        lat = nc_obs.variables["lat"][:,:]
        lon = nc_obs.variables["lon"][:,:]
        nc_obs.close()
        #
        path_output = paths["output"] + self.forecast_start_date[0:4] + "/" + self.forecast_start_date[4:6] + "/"
        if os.path.isdir(path_output) == False:
            os.system("mkdir -p " + path_output)
        #
        filename_output = path_output + "SIC_" + self.forecast_start_date + ".nc"
        output_netcdf = netCDF4.Dataset(filename_output, "w", format = "NETCDF4")
        #
        time = output_netcdf.createDimension("time", len(self.lead_times))
        x = output_netcdf.createDimension("x", len(xc))
        y = output_netcdf.createDimension("y", len(yc))
        #
        time = output_netcdf.createVariable("time", "d", ("time"))
        x = output_netcdf.createVariable("x", "d", ("x"))
        y = output_netcdf.createVariable("y", "d", ("y"))
        SIC = output_netcdf.createVariable("SIC", "d", ("time", "y","x"))
        #
        time.standard_name = "time"
        time.units = "days since the forecast start date"
        x.standard_name = "projection x coordinate"
        x.units = "m"
        y.standard_name = "projection y coordinate"
        y.units = "m"
        SIC.standard_name = "sea ice concentration"
        SIC.units = "%"
        #
        time[:] = self.lead_times
        x[:] = xc
        y[:] = yc
        SIC[:,:,:] = SIC_anomaly_forecasts
        #
        output_netcdf.description = "+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere"
        output_netcdf.close()
    #
    def __call__(self):
        SIC_anomaly_forecasts = self.make_anomaly_forecasts()
        self.write_netCDF(SIC_anomaly_forecasts)


# In[12]:


list_dates = make_list_dates(date_min, date_max)
forecast_start_date = list_dates[$SGE_TASK_ID - 1]
LSM = load_LSM()
#
params_anomaly_forecasts = {"forecast_start_date": forecast_start_date,
                            "lead_times": lead_times,
                            "LSM": LSM,
                            "clim_first_year": clim_first_year,
                            "clim_last_year": clim_last_year,
                            "paths": paths,
                            }
#
make_forecasts = Anomaly_persistence_forecasts(**params_anomaly_forecasts)
make_forecasts()
#
###################################################################################################
EOF

python3 "/home/cyrilp/Documents/PROG/Anomaly_forecasts_""$SGE_TASK_ID"".py"
