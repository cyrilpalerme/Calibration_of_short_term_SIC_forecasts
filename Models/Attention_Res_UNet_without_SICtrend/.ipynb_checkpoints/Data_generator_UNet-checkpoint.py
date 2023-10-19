#!/usr/bin/env python
# coding: utf-8

# In[1]:


import netCDF4
import tensorflow as tf
import numpy as np
tf.keras.utils.set_random_seed(1234)


# Data_generator
#     
#     list_predictors: list of predictors (list format)
#     list_labels: list of labels (list format)
#     list_dates: list of dates (list format)   
#     lead_time: lead time (starting at 0) in integer format
#     standard: dictionary containing the standardization statistics (mean, standard deviation, min, max)
#     batch_size: batch size (integer)
#     path_data: path where the data are located
#     dim: tuple of two dimensions indicating the dimensions of the input dataindicating the dimensions of the input data (y_dim, x_dim) 

# In[2]:


class Data_generator(tensorflow.keras.utils.Sequence):
    def __init__(self, list_predictors, list_labels, list_dates, lead_time, standard, batch_size, path_data, dim, shuffle):
        self.list_predictors = list_predictors
        self.list_labels = list_labels
        self.list_dates = list_dates
        self.lead_time = lead_time
        self.standard = standard
        self.batch_size = batch_size
        self.path_data = path_data
        self.dim = dim
        self.shuffle = shuffle
        self.list_IDs = np.arange(len(list_dates))
        self.n_predictors = len(list_predictors)
        self.n_labels = len(list_labels)
        self.on_epoch_end()
    #
    def __len__(self): # Number of batches per epoch
        return int(np.ceil(len(self.list_IDs)) / self.batch_size)
    #
    def __getitem__(self, index): # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_batch)
        return(X, y)
    #
    def on_epoch_end(self): # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            rng = np.random.default_rng()
            rng.shuffle(self.indexes)
    #
    def standardize(self, var, var_data):
        stand_data = (var_data - self.standard[var + "_mean"]) / self.standard[var + "_std"]
        return(stand_data)
    #
    def normalize(self, var, var_data):
        norm_data = (var_data - self.standard[var + "_min"]) / (self.standard[var + "_max"] - self.standard[var + "_min"])
        return(norm_data)
    #
    def __data_generation(self, list_IDs_batch): # Generates data containing batch_size samples
        #
        # Initialization
        X = np.full((self.batch_size, *self.dim, self.n_predictors), np.nan)
        y = np.full((self.batch_size, *self.dim, self.n_labels), np.nan)
        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            date_ID = self.list_dates[ID]
            file_ID = self.path_data + date_ID[0:4] + "/" + date_ID[4:6] + "/" + "Dataset_" + date_ID + ".nc"
            nc = netCDF4.Dataset(file_ID, "r")
            #
            for v, var in enumerate(self.list_predictors):
                if var == "initial_bias":
                    var_data = nc.variables["TOPAZ_SIC"][0,:,:] - nc.variables["SICobs_SIC"][:,:]
                else:
                    vardim = nc.variables[var].ndim
                    if vardim == 2:
                        var_data = nc.variables[var][:,:]
                    elif vardim == 3:
                        var_data = nc.variables[var][self.lead_time,:,:]
                #
                X[i,:,:,v] = self.normalize(var, var_data)
            #
            for v, var in enumerate(self.list_labels):
                if "_SIE_" in var:
                    y[i,:,:,v] = nc.variables[var][self.lead_time,:,:]
                else: 
                    y[i,:,:,v] = self.normalize(var, nc.variables[var][self.lead_time,:,:])
            #
            nc.close()
        return(X, y)

