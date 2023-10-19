#!/usr/bin/env python
# coding: utf-8

# In[18]:


import tensorflow as tf


# # Residual U-Net

# In[ ]:


class Res_UNet():
    def __init__(self, list_predictors, list_targets, patch_dim, batch_size, n_filters, activation, kernel_initializer, batch_norm, pooling_type, dropout):
        self.list_predictors = list_predictors
        self.list_targets = list_targets
        self.patch_dim = patch_dim
        self.batch_size = batch_size
        self.n_filters = n_filters
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.batch_norm = batch_norm
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.n_predictors = len(list_predictors)
    #
    def residual_conv_block(self, x, n_filters, padding = "same"):
        conv = tf.keras.layers.Conv2D(n_filters, kernel_size = (3,3), padding = padding, kernel_initializer = self.kernel_initializer)(x)
        if self.batch_norm == True:
            conv = tf.keras.layers.BatchNormalization(axis = 3)(conv)
        conv = tf.keras.layers.Activation(self.activation)(conv)
        #
        conv = tf.keras.layers.Conv2D(n_filters, kernel_size = (3,3), padding = padding, kernel_initializer = self.kernel_initializer)(conv)
        if self.batch_norm == True:
            conv = tf.keras.layers.BatchNormalization(axis = 3)(conv)
        #
        shortcut = tf.keras.layers.Conv2D(n_filters, kernel_size = (1,1), padding = padding)(x)
        if self.batch_norm == True:
            shortcut = tf.keras.layers.BatchNormalization(axis = 3)(shortcut)
        #
        res_path = tf.keras.layers.add([shortcut, conv])
        res_path = tf.keras.layers.Activation(self.activation)(res_path)
        #
        return(res_path)
    #
    def downsample_block(self, x, n_filters, pool_size = (2,2), strides = 2):
        f = self.residual_conv_block(x, n_filters)
        #
        if self.pooling_type == "Max":
            p = tf.keras.layers.MaxPool2D(pool_size = pool_size, strides = strides)(f)
        elif self.pooling_type == "Average":
            p = tf.keras.layers.AveragePooling2D(pool_size = pool_size, strides = strides)(f)
        #
        p = tf.keras.layers.Dropout(self.dropout)(p)
        return(f, p)  
    #
    def upsample_block(self, x, conv_features, n_filters, kernel_size = (2,2), strides = 2, padding = "same"):
        x = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size = kernel_size, strides = strides, padding = padding)(x)
        x = tf.keras.layers.concatenate([x, conv_features])
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = self.residual_conv_block(x, n_filters)
        return(x)
    #
    def make_unet_model(self): 
        inputs = tf.keras.layers.Input(shape = (*self.patch_dim, self.n_predictors))
        # Encoder (downsample)
        f1, p1 = self.downsample_block(inputs, self.n_filters[0])
        f2, p2 = self.downsample_block(p1, self.n_filters[1])
        f3, p3 = self.downsample_block(p2, self.n_filters[2])
        f4, p4 = self.downsample_block(p3, self.n_filters[3])
        f5, p5 = self.downsample_block(p4, self.n_filters[4])
        # Bottleneck
        u5 = self.residual_conv_block(p5, self.n_filters[5])
        # Decoder (upsample)
        u4 = self.upsample_block(u5, f5, self.n_filters[4])
        u3 = self.upsample_block(u4, f4, self.n_filters[3])
        u2 = self.upsample_block(u3, f3, self.n_filters[2])
        u1 = self.upsample_block(u2, f2, self.n_filters[1])
        u0 = self.upsample_block(u1, f1, self.n_filters[0])
        # output
        SIC = tf.keras.layers.Conv2D(1, (1, 1), padding = "same", activation = "linear", dtype = tf.float32, name = "SIC")(u0)
        unet_model = tf.keras.Model(inputs, SIC, name = "U-Net")
        #
        return(unet_model)

