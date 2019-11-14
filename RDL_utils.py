# -*- coding: utf-8 -*-
# FILE:           RDL_utils.py
# DATE:           2018
# AUTHOR:         Nick.Nikzad and Aaron Nicolson
# AFFILIATION:    Institute for Integrated and Intelligent Systems, Griffith University, Australia


from __future__ import division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages



## LOSS FUNCTIONS
def loss(target, estimate, loss_fnc):
	'loss functions for gradient descent.'
	epsilon=1e-5
   
	with tf.name_scope(loss_fnc + '_loss'):
		if loss_fnc == 'quadratic':
			loss = tf.reduce_sum(tf.square(tf.subtract(target, estimate)), axis=1)
#		if loss_fnc == 'cross_entropy':
#			loss=tf.negative(tf.reduce_mean(tf.multiply(target, tf.log(estimate+epsilon)))# - tf.multiply((1. - target), tf.log(1. - estimate+epsilon)),axis=1)
		if loss_fnc == 'sigmoid_cross_entropy':
			loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=estimate), axis=1)
		if loss_fnc == 'mse':
			loss = tf.losses.mean_squared_error(labels=target, predictions=estimate)
		if loss_fnc == 'softmax_xentropy':
			loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=estimate)
		if loss_fnc == 'sigmoid_xentropy':
			loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=estimate)
	return loss

## GRADIENT DESCENT OPTIMISERS
def optimizer(loss, lr=None, epsilon=None, var_list=None, optimizer='adam', grad_clip=False):
    'optimizers for training.'
    with tf.name_scope(optimizer + '_opt'):
        if optimizer == 'adam':
            if lr == None: lr = 0.001 # default.
            if epsilon == None: epsilon = 1e-8 # default.
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
        if optimizer == 'nadam':
            if lr == None: lr = 0.001 # default.
            if epsilon == None: epsilon = 1e-8 # default.
            optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, epsilon=epsilon)
        if optimizer == 'sgd':
            if lr == None: lr = 0.5 # default.
            optimizer = tf.train.GradientDescentOptimizer(lr)
        if optimizer == 'moment':
            if lr == None: lr = 0.001 # default.
            nesterov_momentum = 0.9
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, 
                                                   momentum=nesterov_momentum, use_nesterov=True) 
        grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
        grads_and_vars = [(tf.where(tf.is_nan(gv[0]),tf.zeros_like(gv[0]),gv[0]), gv[1]) for gv in grads_and_vars]            
        
        if grad_clip:
            clip_norm=1
            grads_and_vars = [(tf.clip_by_value(gv[0], -10., 10.), gv[1]) for gv in grads_and_vars]
#            optimizer=tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm)
        train_op = optimizer.apply_gradients(grads_and_vars)#optimizer.minimize(loss,name="training_op")#
    return train_op, optimizer


#################### conv2d_BC_relu block   
def conv2d_relu_norm(x,stride,dense_filters,is_training,seq_len,
                   max_drop_rate,name,conv_caus=True,dilation_rate=1,kernel_size=3):

    if kernel_size==1:
        output=block_unit(x, 'BU1',  kernel_size, dense_filters, seq_len, name, is_training,dilation_rate=1, conv_caus=True,padding="valid")
    else:
        output=block_unit(x, 'BU4', kernel_size, dense_filters, seq_len, name, is_training,dilation_rate=dilation_rate, conv_caus=True,padding="valid")


            
    output=tf.layers.dropout(output,rate=max_drop_rate,training=is_training)#np.random.uniform(low=0.0,high=max_drop_rate))
        
    return output


def block_unit(input, block_unit, conv_size, conv_filt, seq_len, unit_id, training,dilation_rate, conv_caus=True,padding="valid"):
	with tf.variable_scope(block_unit + '_' + str(unit_id)):
		if block_unit == 'BU1': U = conv_layer(tf.nn.relu(masked_layer_norm(input, 
			seq_len)), conv_size, conv_filt, seq_len, conv_caus=conv_caus,padding=padding) # (LN -> ReLU -> W).
		elif block_unit == 'BU2': U = conv_layer(tf.nn.relu(masked_batch_norm(input, 
			seq_len, training)), conv_size, conv_filt, seq_len, conv_caus=conv_caus,padding=padding) # (BN -> ReLU -> W).
		elif block_unit == 'BU3': U = tf.nn.relu(masked_layer_norm(conv_layer(input, 
			conv_size, conv_filt, seq_len), seq_len)) # (DCC -> LN -> ReLU). (no dropout, and WeightNorm replaced with LayerNorm).
		elif block_unit == 'BU4': U = conv_layer(tf.nn.relu(masked_layer_norm(input, 
			seq_len)), conv_size, conv_filt, seq_len, conv_caus=conv_caus,padding=padding, dilation_rate=dilation_rate) # (LN -> ReLU -> DCC).
		else: # residual unit does not exist.
			raise ValueError('Residual unit does not exist: %s.' % (block_unit))
		return U

## CNN LAYER
def conv_layer(input, conv_size, conv_filt, seq_len, conv_caus=True,padding="valid", dilation_rate=1, 
	use_bias=True, bias_init=tf.constant_initializer(0.0)):
	if conv_caus:
		input = tf.concat([tf.zeros([tf.shape(input)[0], (conv_size - 1)*dilation_rate, 
			tf.shape(input)[2]]), input], 1)
	conv = tf.layers.conv1d(input, conv_filt, conv_size, dilation_rate=dilation_rate, 
		activation=None, padding=padding, use_bias=use_bias, bias_initializer=bias_init) # 1D CNN: (conv_size, conv_filt).
	conv = tf.multiply(conv, tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 
		2), tf.float32))
	return conv

## MASKED LAYER NORM
def masked_layer_norm(input, seq_len, centre=True, scale=True): # layer norm for 3D tensor.
	with tf.variable_scope('Layer_Norm'):
		mask = tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32) # convert mask to float.
		input_dim = input.get_shape().as_list()[-1] # get number of input dimensions.
		den = tf.multiply(tf.reduce_sum(mask, axis=1, keepdims=True), input_dim) # inverse of the number of input dimensions.
		mean = tf.divide(tf.reduce_sum(tf.multiply(input, mask), axis=[1, 2], keepdims=True), den) # mean over the input dimensions.
		var = tf.divide(tf.reduce_sum(tf.multiply(tf.square(tf.subtract(input, mean)), mask), axis=[1, 2], 
			keepdims = True), den) # variance over the input dimensions.
		if centre:
			beta = tf.get_variable("beta", input_dim, dtype=tf.float32,  
				initializer=tf.constant_initializer(0.0), trainable=True)
		else: beta = tf.constant(np.zeros(input_dim), name="beta", dtype=tf.float32)
		if scale:
			gamma = tf.get_variable("Gamma", input_dim, dtype=tf.float32,  
				initializer=tf.constant_initializer(1.0), trainable=True)
		else: gamma = tf.constant(np.ones(input_dim), name="Gamma", dtype=tf.float32)
		norm = tf.nn.batch_normalization(input, mean, var, offset=beta, scale=gamma, 
			variance_epsilon = 1e-10) # normalise batch.
		norm = tf.multiply(norm, mask)
		return norm
## MASKED MINI-BATCH STATISTICS
def masked_batch_stats(input, mask, axes, scaler):
	num = tf.reduce_sum(tf.multiply(input, mask), axis=axes)
	den = tf.multiply(tf.reduce_sum(mask, axis=axes), scaler)
	mean = tf.divide(num, den)
	num = tf.reduce_sum(tf.multiply(tf.square(tf.subtract(input, mean)), 
		mask), axis=axes)
	var = tf.divide(num, den)
	return mean, var

def masked_batch_norm(inp, seq_len, training=False, decay=0.99, centre=True, scale=True):
	with tf.variable_scope('Batch_Norm'):
		mask = tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32)
		input_dim = inp.get_shape().as_list()[-1:]
		moving_mean = tf.get_variable("moving_mean", input_dim, dtype=tf.float32,  
			initializer=tf.zeros_initializer, trainable=False)
		moving_var = tf.get_variable("moving_var", input_dim, dtype=tf.float32,  
			initializer=tf.constant_initializer(1), trainable=False)
		batch_mean, batch_var = masked_batch_stats(inp, mask, [0, 1], 1)
		def update_moving_stats():
			update_moving_mean = moving_averages.assign_moving_average(moving_mean, 
				batch_mean, decay, zero_debias=True)
			update_moving_variance = moving_averages.assign_moving_average(moving_var, 
				batch_var, decay, zero_debias=False)
			with tf.control_dependencies([update_moving_mean, update_moving_variance]):
				return tf.identity(batch_mean), tf.identity(batch_var)
		mean, var = tf.cond(training, true_fn = update_moving_stats,
			false_fn = lambda: (moving_mean, moving_var))
		variance_epsilon = 1e-12       

		if centre:
			beta = tf.get_variable("beta", input_dim, dtype=tf.float32,  
				initializer=tf.constant_initializer(0.0), trainable=True)
		else: beta = tf.constant(np.zeros(input_dim), name="beta", dtype=tf.float32)
		if scale:
			gamma = tf.get_variable("Gamma", input_dim, dtype=tf.float32,  
				initializer=tf.constant_initializer(1.0), trainable=True)
		else: gamma = tf.constant(np.ones(input_dim), name="Gamma", dtype=tf.float32)
		norm = tf.nn.batch_normalization(inp, mean, var, beta, gamma, 
			variance_epsilon)
		norm = tf.multiply(norm, mask)
		return norm
############################# Aux functions
    
def shortcut_projection(x,in_filters,out_filters,name):
    imapped_x=tf.layers.dense(x, out_filters, use_bias = False)
    
    return imapped_x


def x_scale (x, newshape,name, keep_ch=True):
    new_ch=int(newshape[2])    
    x_ch=int(np.shape(x)[2])
    if ((keep_ch==False) and (x_ch!=new_ch)):
        x= shortcut_projection(x,x_ch,new_ch,name)       
    return x
    
def concat_x1x2 (x1,x2):
    return  tf.concat([x1,x2],axis=2)


