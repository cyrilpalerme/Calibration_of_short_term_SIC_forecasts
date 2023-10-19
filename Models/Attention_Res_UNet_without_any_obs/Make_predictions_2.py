#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import time
import h5py
import netCDF4
import datetime
import numpy as np
import tensorflow as tf
#
tf.keras.mixed_precision.set_global_policy("mixed_float16")
print("GPUs available: ", tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#
t0 = time.time()


# # Constants

# In[ ]:


experiment_name = "SIC_Attention_Res_UNet_without_any_obs"
lead_time = 2
#
function_path = "/lustre/storeB/users/cyrilp/COSI/Scripts/Models/" + experiment_name + "/"
sys.path.insert(0, function_path)
from Data_generator_UNet import *
from Attention_Res_UNet import *
#
date_min_test = "20220101"
date_max_test = "20221231"
#
paths = {}
paths["SDAP"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/SDAP/OSISAF/Without_coastlines/Gaussian_filter_0km/SDAP_2012_2021/"
paths["anomaly_persistence"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Anomaly_persistence_forecasts/AMSR2/"
paths["training"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Training/Land_free_ocean/"
paths["standard"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Training/Standardization/"
paths["model_weights"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Model_weights/" + experiment_name + "/"
paths["ice_edge_lengths"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/AMSR2_ice_edge_length/"
paths["predictions_netCDF"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Predictions/" + experiment_name + "/lead_time_" + str(lead_time) + "_days/netCDF/"
paths["prediction_scores"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Predictions/" + experiment_name + "/lead_time_" + str(lead_time) + "_days/scores/"
#
for var in paths:
    if os.path.isdir(paths[var]) == False:
        os.system("mkdir -p " + paths[var])
#
file_standardization = paths["standard"] + "Stats_standardization_20130103_20201231_weekly.h5"
file_model_weights = paths["model_weights"] + "UNet_leadtime_" + str(lead_time) + "_days.h5"
#
grid_cell_area = 12500 ** 2  # m2


# # U-Net parameters

# In[ ]:


list_predictors = ["LSM", "TOPAZ_SIC", "ECMWF_T2M_cum", "ECMWF_wind_x_cum", "ECMWF_wind_y_cum"]
list_targets = ["TARGET_AMSR2_SIC"]
#
model_params = {"list_predictors": list_predictors,
                "list_targets": list_targets, 
                "patch_dim": (544, 544),
                "batch_size": 4,
                "n_filters": [32, 64, 128, 256, 512, 1024],
                "activation": "relu",
                "kernel_initializer": "he_normal",
                "batch_norm": True,
                "pooling_type": "Average",
                "dropout": 0,
                }


# # Load Land-Sea mask
# 
#     1: Ocean
#     0: Land

# In[ ]:


def load_land_sea_mask(paths = paths, start_date = "20180104"):
    filename = paths["training"] + start_date[0:4] + "/" + start_date[4:6] + "/" + "Dataset_" + start_date + ".nc"
    nc = netCDF4.Dataset(filename, "r")
    LSM = nc.variables["LSM"][:,:] 
    nc.close()
    return(LSM)


# # make_list_dates function
# 
#     date_min: earliest date of the period ("YYYYMMDD")
#     date_max: latest date of the period ("YYYYMMDD")
#     frequency: "daily" or "weekly"
#     path_data: path where the data are stored

# In[ ]:


def make_list_dates(date_min, date_max, frequency, path_data, lead_time = lead_time):
    current_date = datetime.datetime.strptime(date_min, '%Y%m%d')
    end_date = datetime.datetime.strptime(date_max, '%Y%m%d')
    list_dates = []
    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        filename = path_data + date_str[0:4] + "/" + date_str[4:6] + "/" + "Dataset_" + date_str + ".nc"
        if os.path.isfile(filename):
            nc = netCDF4.Dataset(filename, "r")
            TARGET_AMSR2_SIC = nc.variables["TARGET_AMSR2_SIC"][lead_time,:,:]
            nc.close()
            if np.sum(np.isnan(TARGET_AMSR2_SIC)) == 0:
                list_dates.append(date_str)
        #
        if frequency == "daily":
            current_date = current_date + datetime.timedelta(days = 1)
        elif frequency == "weekly":
            current_date = current_date + datetime.timedelta(days = 7)
    return(list_dates)


# # Standardization data

# In[ ]:


def load_standardization_data(file_standardization):
    standard = {}
    hf = h5py.File(file_standardization, "r")
    for var in hf:
        if "ECMWF" in var:
            standard[var] = np.array(hf[var])[lead_time]
        else:
            standard[var] = hf[var][()]
    hf.close()
    return(standard)


# # Extract evaluation data

# In[ ]:


def extract_eval_data(start_date, lead_time):
    previous_day = (datetime.datetime.strptime(start_date, "%Y%m%d") - datetime.timedelta(days = 1)).strftime("%Y%m%d")
    #
    Eval_data = {}
    filename_training = paths["training"] + start_date[0:4] + "/" + start_date[4:6] + "/" + "Dataset_" + start_date + ".nc"
    nc_training = netCDF4.Dataset(filename_training, "r")
    #
    for var in ["x", "y", "lat", "lon", "TOPAZ_SIC", "TARGET_AMSR2_SIC", "TARGET_AMSR2_SIE_10", "TARGET_AMSR2_SIE_20", "SICobs_AMSR2_SIC", "TOPAZ_bias_corrected"]:
        if var == "TOPAZ_bias_corrected":
            ini_bias = nc_training.variables["TOPAZ_SIC"][0,:,:] - nc_training.variables["SICobs_AMSR2_SIC"][:,:]
            T4_SIC = nc_training.variables["TOPAZ_SIC"][lead_time,:,:]
            T4_SIC_bias_corrected = T4_SIC - ini_bias
            T4_SIC_bias_corrected[T4_SIC_bias_corrected > 100] = 100 
            T4_SIC_bias_corrected[T4_SIC_bias_corrected < 0] = 0
            Eval_data[var] = np.copy(T4_SIC_bias_corrected)
        else:
            if nc_training.variables[var].ndim == 1:
                Eval_data[var] = nc_training.variables[var][:]
            elif nc_training.variables[var].ndim == 2:
                Eval_data[var] = nc_training.variables[var][:,:]
            elif nc_training.variables[var].ndim == 3:
                Eval_data[var] = nc_training.variables[var][lead_time,:,:]
    nc_training.close()
    Eval_data["TARGET_AMSR2_SIE_15"] = np.zeros(np.shape(Eval_data["TARGET_AMSR2_SIC"]))
    Eval_data["TARGET_AMSR2_SIE_15"][Eval_data["TARGET_AMSR2_SIC"] >= 15] = 1
    #
    filename_anomaly_persistence = paths["anomaly_persistence"] + previous_day[0:4] + "/" + previous_day[4:6] + "/" + "SIC_" + previous_day + ".nc"
    nc_anomaly_persistence = netCDF4.Dataset(filename_anomaly_persistence, "r")
    Eval_data["Anomaly_persistence_SIC"] = nc_anomaly_persistence.variables["SIC"][lead_time + 1,:,:]
    nc_anomaly_persistence.close()
    #
    target_date = (datetime.datetime.strptime(start_date, "%Y%m%d") + datetime.timedelta(days = lead_time)).strftime("%Y%m%d")
    file_ice_edge_lengths = paths["ice_edge_lengths"] + target_date[0:4] + "/" + target_date[4:6] + "/" + "Ice_edge_lengths_" + target_date + ".h5"
    hf = h5py.File(file_ice_edge_lengths, "r")
    for var in hf:
        Eval_data[var] = np.array(hf[var])
    hf.close()    
    #
    if int(start_date[0:4]) >= 2022:
        filename_SDAP = paths["SDAP"] + previous_day[0:4] + "/" + previous_day[4:6] + "/" + "SDAP_" + previous_day + ".nc"
        nc_SDAP = netCDF4.Dataset(filename_SDAP, "r")
        Eval_data["SDAP10"] = nc_SDAP.variables["SDAP10"][lead_time + 1,:,:]
        Eval_data["SDAP15"] = nc_SDAP.variables["SDAP15"][lead_time + 1,:,:]
        Eval_data["SDAP20"] = nc_SDAP.variables["SDAP20"][lead_time + 1,:,:]
        nc_SDAP.close()
    #
    return(Eval_data)


# # Save predictions in netCDF file

# In[ ]:


def save_predictions_in_netCDF(start_date, Pred_data, Eval_data, paths = paths):
    file_output = paths["predictions_netCDF"] + "Predictions_" + start_date + ".nc"
    output_netcdf = netCDF4.Dataset(file_output, 'w', format = 'NETCDF4')
    Outputs = vars()
    #
    dimensions = ["x", "y"]
    for di in dimensions:
        Outputs[di] = output_netcdf.createDimension(di, len(Eval_data[di]))
    #
    dim_variables = dimensions + ["lat", "lon"]
    for dv in dim_variables:
        if Eval_data[dv].ndim == 1:
            Outputs[dv] = output_netcdf.createVariable(dv, "d", (dv))
            Outputs[dv][:] = Eval_data[dv]   
            if dv == "x" or dv == "y":
                Outputs[dv].standard_name = "projection_" + dv + "_coordinate"
                Outputs[dv].units = "m"
        elif Eval_data[dv].ndim == 2:
            Outputs[dv] = output_netcdf.createVariable(dv, "d", ("y", "x"))
            Outputs[dv][:,:] = Eval_data[dv]
            if dv == "lat":
                Outputs[dv].standard_name = "latitude"
            elif dv == "lon":
                Outputs[dv].standard_name = "longitude"
            Outputs[dv].units = "degrees"
    #
    for var in Eval_data:
        if "Ice_edge_lengths_SIC" in var:
            pass
        else:
            if (var in dim_variables) == False:
                Outputs[var] = output_netcdf.createVariable(var, "d", ("y", "x"))
                Outputs[var][:,:] = Eval_data[var]
                if "SIC" in var:
                    Outputs[var].standard_name = "sea ice concentration"
                    Outputs[var].units = "%"
                elif "SIE" in var:
                    Outputs[var].standard_name = "sea ice extent"
                    Outputs[var].units = "1 if sea ice concentration higher than " + var[-2:len(var)] + " %, 0 otherwise"
                elif "TOPAZ_bias_corrected" in var:
                    Outputs[var].standard_name = "TOPAZ bias corrected forecasts"
                    Outputs[var].units = "%"
                elif "SDAP" in var:
                    Outputs[var].standard_name = "Spatial damped anomaly persistence forecasts (SDAP) for sea-ice concentration higher than " + var[4:6] + " %"
                    Outputs[var].units = "Fraction"
    #
    for var in Pred_data:
        Outputs[var] = output_netcdf.createVariable(var, "d", ("y", "x"))
        Outputs[var][:,:] = Pred_data[var]
        if "TOPAZ_error" in var:
            Outputs[var].standard_name = "Predicted SIC from model error"
            Outputs[var].units = "%"
        elif "SIC" in var:
            Outputs[var].standard_name = var
            Outputs[var].units = "%" 
        elif "SIE" in var:
            Outputs[var].standard_name = "Predicted sea ice extent (SIC threshold " + var[-2:len(var)] + " %)"
            Outputs[var].units = "1: ice, 0: ice-free"   
    #
    output_netcdf.close()


# # Functions for calculating SIC from predictions, and for making binary classification

# In[ ]:


def SIC_from_normalized_SIC(variable_name, field, standard):
    Predicted_SIC = field * (standard[variable_name + "_max"] - standard[variable_name + "_min"]) + standard[variable_name + "_min"]
    Predicted_SIC[Predicted_SIC > 100] = 100
    Predicted_SIC[Predicted_SIC < 0] = 0
    return(Predicted_SIC)
#
def SIC_from_normalized_model_error(variable_name, field, standard, si_model_SIC):
    model_error = field * (standard[variable_name + "_max"] - standard[variable_name + "_min"]) + standard[variable_name + "_min"]
    print("model_error", np.min(model_error), np.max(model_error), np.mean(model_error), np.median(model_error))
    Predicted_SIC = si_model_SIC - model_error
    Predicted_SIC[Predicted_SIC > 100] = 100
    Predicted_SIC[Predicted_SIC < 0] = 0
    return(Predicted_SIC)
#
def binary_classification(field, threshold):
    output = np.zeros(np.shape(field))
    output[field > threshold] = 1
    return(output)
#
def RMSE(SIC_forecasts, SIC_observations, LSM):
    SIC_forecasts = np.ndarray.flatten(SIC_forecasts[LSM == 1])
    SIC_observations = np.ndarray.flatten(SIC_observations[LSM == 1])
    MSE = np.sum((SIC_forecasts - SIC_observations) ** 2) / len(SIC_observations)
    RMSE = np.sqrt(MSE)
    return(RMSE)


# # Metrics

# In[ ]:


def load_ice_edge_lengths(file_ice_edge_lengths):
    Dataset = {}
    hf = h5py.File(file_ice_edge_lengths, "r")
    for var in hf:
        Dataset[var] = np.array(hf[var])
    hf.close()
    return(Dataset)
#
def IIEE(SIE_obs, SIE_forecast, grid_cell_area):
    Flag_SIE = np.full(np.shape(SIE_obs), np.nan)
    Flag_SIE[SIE_forecast == SIE_obs] = 0
    Flag_SIE[SIE_forecast < SIE_obs] = -1
    Flag_SIE[SIE_forecast > SIE_obs] = 1
    Underestimation = np.sum(Flag_SIE == -1) * grid_cell_area
    Overestimation = np.sum(Flag_SIE == 1) * grid_cell_area
    IIEE_metric = Underestimation + Overestimation
    return(IIEE_metric, Underestimation, Overestimation)
#
def SPS(SIP_obs, SIP_forecast, grid_cell_area):
    SPS_metric = np.nansum(grid_cell_area * (SIP_forecast - SIP_obs)**2)
    return(SPS_metric)


# In[ ]:


def verification_scores(Pred_data, Eval_data, start_date, LSM, grid_cell_area = grid_cell_area):
    day_of_year = int(datetime.datetime.strptime(start_date, "%Y%m%d").strftime('%j'))
    #
    ML = {}
    ML["SIE_10"] = binary_classification(Pred_data["Predicted_SIC"], 10)
    ML["SIE_15"] = binary_classification(Pred_data["Predicted_SIC"], 15)
    ML["SIE_20"] = binary_classification(Pred_data["Predicted_SIC"], 20)
    #
    ML_post_processed = {}
    ML_post_processed["SIC"] = np.copy(Pred_data["Predicted_SIC"])
    ML_post_processed["SIC"][Pred_data["Predicted_SIC"] <= 2] = 0
    #
    Persistence = {}
    Persistence["SIE_10"] = binary_classification(Eval_data["SICobs_AMSR2_SIC"], 10)
    Persistence["SIE_15"] = binary_classification(Eval_data["SICobs_AMSR2_SIC"], 15)
    Persistence["SIE_20"] = binary_classification(Eval_data["SICobs_AMSR2_SIC"], 20)
    #
    TOPAZ = {}
    TOPAZ["SIE_10"] = binary_classification(Eval_data["TOPAZ_SIC"], 10)
    TOPAZ["SIE_15"] = binary_classification(Eval_data["TOPAZ_SIC"], 15)
    TOPAZ["SIE_20"] = binary_classification(Eval_data["TOPAZ_SIC"], 20)
    #
    TOPAZ_bias_corrected = {}
    TOPAZ_bias_corrected["SIE_10"] = binary_classification(Eval_data["TOPAZ_bias_corrected"], 10)
    TOPAZ_bias_corrected["SIE_15"] = binary_classification(Eval_data["TOPAZ_bias_corrected"], 15)
    TOPAZ_bias_corrected["SIE_20"] = binary_classification(Eval_data["TOPAZ_bias_corrected"], 20)
    #
    Anomaly_persistence = {}
    Anomaly_persistence["SIE_10"] = binary_classification(Eval_data["Anomaly_persistence_SIC"], 10)
    Anomaly_persistence["SIE_15"] = binary_classification(Eval_data["Anomaly_persistence_SIC"], 15)
    Anomaly_persistence["SIE_20"] = binary_classification(Eval_data["Anomaly_persistence_SIC"], 20)
    #
    if int(start_date[0:4]) >= 2022:
        SDAP = {}
        SDAP["SIE_10"] = binary_classification(Eval_data["SDAP10"], 0.5)
        SDAP["SIE_15"] = binary_classification(Eval_data["SDAP15"], 0.5)
        SDAP["SIE_20"] = binary_classification(Eval_data["SDAP20"], 0.5)
    #
    Metrics = {}
    Metrics["start_date"] = start_date
    Metrics["Ice_edge_length_SIC10"] = Eval_data["Ice_edge_lengths_SIC10"]
    Metrics["Ice_edge_length_SIC15"] = Eval_data["Ice_edge_lengths_SIC15"]
    Metrics["Ice_edge_length_SIC20"] = Eval_data["Ice_edge_lengths_SIC20"]
    #
    Metrics["RMSE_ML"] = RMSE(Pred_data["Predicted_SIC"], Eval_data["TARGET_AMSR2_SIC"], LSM)
    Metrics["RMSE_ML_post_processed"] = RMSE(ML_post_processed["SIC"], Eval_data["TARGET_AMSR2_SIC"], LSM)
    Metrics["RMSE_TOPAZ"] = RMSE(Eval_data["TOPAZ_SIC"], Eval_data["TARGET_AMSR2_SIC"], LSM)
    Metrics["RMSE_TOPAZ_bias_corrected"] = RMSE(Eval_data["TOPAZ_bias_corrected"], Eval_data["TARGET_AMSR2_SIC"], LSM)
    Metrics["RMSE_Persistence"] = RMSE(Eval_data["SICobs_AMSR2_SIC"], Eval_data["TARGET_AMSR2_SIC"], LSM)
    Metrics["RMSE_Anomaly_persistence"] =  RMSE(Eval_data["Anomaly_persistence_SIC"], Eval_data["TARGET_AMSR2_SIC"], LSM)
    #
    Metrics["IIEElength_10_ML"] = IIEE(Eval_data["TARGET_AMSR2_SIE_10"], ML["SIE_10"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC10"]
    Metrics["IIEElength_10_TOPAZ"] = IIEE(Eval_data["TARGET_AMSR2_SIE_10"], TOPAZ["SIE_10"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC10"]
    Metrics["IIEElength_10_TOPAZ_bias_corrected"] = IIEE(Eval_data["TARGET_AMSR2_SIE_10"], TOPAZ_bias_corrected["SIE_10"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC10"]
    Metrics["IIEElength_10_Persistence"] = IIEE(Eval_data["TARGET_AMSR2_SIE_10"], Persistence["SIE_10"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC10"]  
    Metrics["IIEElength_10_Anomaly_persistence"] = IIEE(Eval_data["TARGET_AMSR2_SIE_10"], Anomaly_persistence["SIE_10"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC10"]
    if int(start_date[0:4]) >= 2022:
        Metrics["IIEElength_10_SDAP"] = IIEE(Eval_data["TARGET_AMSR2_SIE_10"], SDAP["SIE_10"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC10"] 
    #
    Metrics["IIEElength_15_ML"] = IIEE(Eval_data["TARGET_AMSR2_SIE_15"], ML["SIE_15"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC15"]
    Metrics["IIEElength_15_TOPAZ"] = IIEE(Eval_data["TARGET_AMSR2_SIE_15"], TOPAZ["SIE_15"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC15"]
    Metrics["IIEElength_15_TOPAZ_bias_corrected"] = IIEE(Eval_data["TARGET_AMSR2_SIE_15"], TOPAZ_bias_corrected["SIE_15"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC15"]
    Metrics["IIEElength_15_Persistence"] = IIEE(Eval_data["TARGET_AMSR2_SIE_15"], Persistence["SIE_15"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC15"]  
    Metrics["IIEElength_15_Anomaly_persistence"] = IIEE(Eval_data["TARGET_AMSR2_SIE_15"], Anomaly_persistence["SIE_15"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC15"]
    if int(start_date[0:4]) >= 2022:
        Metrics["IIEElength_15_SDAP"] = IIEE(Eval_data["TARGET_AMSR2_SIE_15"], SDAP["SIE_15"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC15"]   
    #
    Metrics["IIEElength_20_ML"] = IIEE(Eval_data["TARGET_AMSR2_SIE_20"], ML["SIE_20"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC20"]
    Metrics["IIEElength_20_TOPAZ"] = IIEE(Eval_data["TARGET_AMSR2_SIE_20"], TOPAZ["SIE_20"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC20"]
    Metrics["IIEElength_20_TOPAZ_bias_corrected"] = IIEE(Eval_data["TARGET_AMSR2_SIE_20"], TOPAZ_bias_corrected["SIE_20"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC20"]
    Metrics["IIEElength_20_Persistence"] = IIEE(Eval_data["TARGET_AMSR2_SIE_20"], Persistence["SIE_20"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC20"]
    Metrics["IIEElength_20_Anomaly_persistence"] = IIEE(Eval_data["TARGET_AMSR2_SIE_20"], Anomaly_persistence["SIE_20"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC20"]
    if int(start_date[0:4]) >= 2022:
        Metrics["IIEElength_20_SDAP"] = IIEE(Eval_data["TARGET_AMSR2_SIE_20"], SDAP["SIE_20"], grid_cell_area)[0] / Metrics["Ice_edge_length_SIC20"]
    #
    for var in Metrics:
        if var != "start_date":
            if "RMSE" in var:
                Metrics[var] = np.round(Metrics[var], 3)
            else:
                Metrics[var] = np.round(Metrics[var])
    #
    return(Metrics)


# # Write_scores function

# In[ ]:


def save_scores(Metrics, paths = paths):
    header = ""
    scores = ""
    for var in Metrics:
        header = header + "\t" + var   
        scores = scores + "\t" + str(Metrics[var]) 
    #
    output_file = paths["prediction_scores"] + "Scores_" + date_min_test + "_" + date_max_test + ".txt"
    if start_date == date_min_test:
        if os.path.isfile(output_file) == True:
            os.system("rm " + output_file)
    #
    if os.path.isfile(output_file) == False:
        output = open(output_file, 'a')
        output.write(header + "\n")
        output.close()
    #
    output = open(output_file, 'a')
    output.write(scores + "\n")
    output.close()


# # Make predictions functions

# In[ ]:


def make_predictions(start_date, model, standard, LSM):
    Eval_data = extract_eval_data(start_date, lead_time)
    #
    params_test = {"list_predictors": model_params["list_predictors"],
                    "list_labels": model_params["list_targets"],
                    "list_dates": [start_date],
                    "lead_time": lead_time,
                    "standard": standard,
                    "batch_size": 1,
                    "path_data": paths["training"],
                    "dim": model_params["patch_dim"],
                    "shuffle": False,
                    }
    #
    test_generator = Data_generator(**params_test)
    predictions = np.squeeze(model.predict(test_generator))
    predictions = SIC_from_normalized_SIC("TARGET_AMSR2_SIC", predictions, standard)
    predictions[:,:][LSM == 0] = 0
    # 
    Pred_data = {}
    Pred_data["Predicted_SIC"] = np.copy(predictions[:,:])
    #
    return(Pred_data, Eval_data)


# # Data processing

# In[ ]:


LSM = load_land_sea_mask()
standard = load_standardization_data(file_standardization)
list_dates_test = make_list_dates(date_min_test, date_max_test, frequency = "daily", path_data = paths["training"])
#
unet_model = Att_Res_UNet(**model_params).make_unet_model()
unet_model.load_weights(file_model_weights)
#print(unet_model.summary())
#
Scores = {}
for sd, start_date in enumerate(list_dates_test):
    print("forecast start_date", start_date)
    try:
        Pred_data, Eval_data = make_predictions(start_date = start_date, model = unet_model, standard = standard, LSM = LSM)
        save_predictions_in_netCDF(start_date, Pred_data, Eval_data)
        Metrics = verification_scores(Pred_data, Eval_data, start_date, LSM, grid_cell_area = grid_cell_area)
        save_scores(Metrics, paths = paths)
    except:
        pass
#
t1 = time.time()
dt = t1 - t0
#
print("Predictions made ! Time: ", dt)

