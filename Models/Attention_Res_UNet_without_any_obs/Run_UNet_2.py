#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import glob
import time
import h5py
import json
import pickle
import netCDF4
import datetime
import numpy as np
import tensorflow as tf
#
tf.keras.utils.set_random_seed(420)
tf.keras.mixed_precision.set_global_policy("mixed_float16")
print("GPUs available: ", tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#
t0 = time.time()


# # Constants

# In[6]:


experiment_name = "SIC_Attention_Res_UNet_without_any_obs"
lead_time = 2
#
function_path = "/lustre/storeB/users/cyrilp/COSI/Scripts/Models/" + experiment_name + "/"
sys.path.insert(0, function_path)
from Data_generator_UNet import *
from Attention_Res_UNet import *
#
date_min_train = "20130103"
date_max_train = "20201231"
date_min_valid = "20210101"
date_max_valid = "20211231"
#
paths = {}
paths["data"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Training/Land_free_ocean/"
paths["standard"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Training/Standardization/"
paths["output"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Model_weights/" + experiment_name + "/"
paths["checkpoints"] = "/lustre/storeB/project/copernicus/cosi/WP3/Data/Model_weights/" + experiment_name + "/Checkpoints/"
#
for var in paths:
    if os.path.isdir(paths[var]) == False:
        os.system("mkdir -p " + paths[var])
#
file_standardization = paths["standard"] + "Stats_standardization_20130103_20201231_weekly.h5"
file_checkpoints = paths["checkpoints"] + "Checkpoints.h5"
#
if os.path.isfile(file_checkpoints) == True:
    os.system("rm " + file_checkpoints)


# # U-Net parameters

# In[2]:


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
#
compile_params = {"initial_learning_rate": 0.005, 
                  "decay_steps": 2550,
                  "decay_rate": 0.5,
                  "staircase": True,
                  "n_epochs": 100,
                  }
#
model_and_compile_params = {**model_params, **compile_params}


# # save_model_parameters function

# In[ ]:


def save_model_parameters(lead_time, model_and_compile_params, model_history, paths = paths):
    file_model_parameters = paths["output"] + "Model_parameters_" + str(lead_time) + "_days.txt"
    file_model_training_history = paths["output"] + "Training_history_leadtime_" + str(lead_time) + "_days.pkl"
    #
    if os.path.isfile(file_model_parameters) == True:
        os.system("rm " + file_model_parameters)
    if os.path.isfile(file_model_training_history) == True:
        os.system("rm " + file_model_training_history)
    #
    pickle.dump(model_history.history, open(file_model_training_history, "wb"))
    with open(file_model_parameters, "w") as output_file:
        output_file.write(json.dumps(model_and_compile_params))


# # make_list_dates function
# 
#     date_min: earliest date of the period ("YYYYMMDD")
#     date_max: latest date of the period ("YYYYMMDD")
#     frequency: "daily" or "weekly"
#     path_data: path where the data are stored

# In[8]:


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

# In[9]:


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


# # Data generator 

# In[10]:


standard = load_standardization_data(file_standardization)
#
list_dates_train = make_list_dates(date_min_train, date_max_train, frequency = "weekly", path_data = paths["data"], lead_time = lead_time)
list_dates_valid = make_list_dates(date_min_valid, date_max_valid, frequency = "daily", path_data = paths["data"], lead_time = lead_time)
#
params_train = {"list_predictors": model_params["list_predictors"],
                "list_labels": model_params["list_targets"],
                "list_dates": list_dates_train,
                "lead_time": lead_time,
                "standard": standard,
                "batch_size": model_params["batch_size"],
                "path_data": paths["data"],
                "dim": model_params["patch_dim"],
                "shuffle": True,
                }
#
params_valid = {"list_predictors": model_params["list_predictors"],
                "list_labels": model_params["list_targets"],
                "list_dates": list_dates_valid,
                "lead_time": lead_time,
                "standard": standard,
                "batch_size": model_params["batch_size"],
                "path_data": paths["data"],
                "dim": model_params["patch_dim"],
                "shuffle": True,
                }
#
train_generator = Data_generator(**params_train)
valid_generator = Data_generator(**params_valid)


# # Data processing

# In[ ]:


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = compile_params["initial_learning_rate"],
    decay_steps = compile_params["decay_steps"],
    decay_rate = compile_params["decay_rate"],
    staircase = compile_params["staircase"])
#
opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
loss = {"SIC": tf.keras.losses.MeanSquaredError()}
metrics = {"SIC": tf.keras.metrics.RootMeanSquaredError()}
#
unet_model = Att_Res_UNet(**model_params).make_unet_model()
print(type(unet_model))
print(unet_model.summary())
unet_model.compile(loss = loss, metrics = metrics, optimizer = opt)
print("Model compiled")
#
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = file_checkpoints, save_weights_only = True, monitor = 'val_loss', mode = 'min', verbose = 2, save_best_only = True)
#
model_history = unet_model.fit(train_generator, validation_data = valid_generator, epochs = compile_params["n_epochs"], verbose = 2, callbacks = [checkpoint])
print("Model fitted")
#
filename_model = 'UNet_leadtime_' + str(lead_time) + '_days.h5'
unet_model.save_weights(paths["output"] + filename_model)
#
save_model_parameters(lead_time, model_params, model_history)
#
t1 = time.time()
dt = t1 - t0
print("Computing time: " + str(dt) + " seconds")

