#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.keras.utils.set_random_seed(1234)


# In[1]:


class Att_UNet():
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
    def repeat_elem(self, tensor, rep):
        return tf.keras.layers.Lambda(lambda x, repnum: tf.keras.backend.repeat_elements(x, repnum, axis = 3), arguments = {'repnum': rep})(tensor)
    #
    def gating_signal(self, x, n_filters, batch_norm = False):
        x = tf.keras.layers.Conv2D(n_filters, (1,1), padding = "same")(x)
        if batch_norm == True:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return(x)
    #
    def attention_block(self, x, g, inter_shape):
        shape_x = tf.keras.backend.int_shape(x)
        shape_g = tf.keras.backend.int_shape(g)
        #
        theta_x = tf.keras.layers.Conv2D(inter_shape, kernel_size = (2,2), strides = (2,2), padding = "same")(x) 
        shape_theta_x = tf.keras.backend.int_shape(theta_x)
        #
        phi_g = tf.keras.layers.Conv2D(inter_shape, kernel_size = (1,1), padding = "same")(g)
        upsample_g = tf.keras.layers.Conv2DTranspose(inter_shape, (3,3), 
                                                     strides = (shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                                     padding = "same")(phi_g)
        
        concat_xg = tf.keras.layers.add([upsample_g, theta_x])
        act_xg = tf.keras.layers.Activation("relu")(concat_xg)
        #
        psi = tf.keras.layers.Conv2D(1, (1,1), padding = "same")(act_xg)
        sigmoid_xg = tf.keras.layers.Activation("sigmoid")(psi)
        shape_sigmoid = tf.keras.backend.int_shape(sigmoid_xg)
        #
        upsample_psi = tf.keras.layers.UpSampling2D(size = (shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)
        upsample_psi = self.repeat_elem(upsample_psi, shape_x[3])
        y = tf.keras.layers.multiply([upsample_psi, x])
        #
        result = tf.keras.layers.Conv2D(shape_x[3], (1,1), padding = "same")(y)
        result_bn = tf.keras.layers.BatchNormalization()(result)
        #
        return(result_bn)
    #
    def conv_block(self, x, n_filters, padding = "same"):
        x = tf.keras.layers.Conv2D(n_filters, kernel_size = (3,3), padding = padding, kernel_initializer = self.kernel_initializer)(x)
        if self.batch_norm == True:
            x = tf.keras.layers.BatchNormalization(axis = 3)(x)
        x = tf.keras.layers.Activation(self.activation)(x)
        #
        x = tf.keras.layers.Conv2D(n_filters, kernel_size = (3,3), padding = padding, kernel_initializer = self.kernel_initializer)(x)
        if self.batch_norm == True:
            x = tf.keras.layers.BatchNormalization(axis = 3)(x)
        x = tf.keras.layers.Activation(self.activation)(x)
        #
        return(x)
    #
    def downsample_block(self, x, n_filters, pool_size = (2,2), strides = 2):
        f = self.conv_block(x, n_filters)
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
        gating = self.gating_signal(x, n_filters)
        att = self.attention_block(conv_features, gating, n_filters)
        up_att = tf.keras.layers.UpSampling2D(size = (2, 2), data_format = "channels_last")(x)
        up_att = tf.keras.layers.concatenate([up_att, att], axis = 3)
        up_conv = self.conv_block(up_att, n_filters)
        return(up_conv)
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
        u5 = self.conv_block(p5, self.n_filters[5])
        # Decoder (upsample)
        u4 = self.upsample_block(u5, f5, self.n_filters[4])
        u3 = self.upsample_block(u4, f4, self.n_filters[3])
        u2 = self.upsample_block(u3, f3, self.n_filters[2])
        u1 = self.upsample_block(u2, f2, self.n_filters[1])
        u0 = self.upsample_block(u1, f1, self.n_filters[0])
        # outputs
        SICerror = tf.keras.layers.Conv2D(1, (1, 1), padding = "same", activation = "linear", dtype = tf.float32, name = "SICerror")(u0)
        unet_model = tf.keras.Model(inputs, SICerror, name = "U-Net")
        #
        return(unet_model)

