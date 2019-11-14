# -*- coding: utf-8 -*-
# FILE:           RDL_network.py
# DATE:           2018
# AUTHOR:         Nick.Nikzad
# AFFILIATION:    Institute for Integrated and Intelligent Systems, Griffith University, Australia
# BRIEF:          'Residual-Dense Lattice architecture.

import numpy as np
import tensorflow as tf
from RDL_utils import x_scale,concat_x1x2,conv2d_relu_norm


def RDL_Net(input_x,num_outputs,net_height,growth_rate,no_layers_level,
           max_drop,is_training,seq_len,net_blocks,no_level):
    
    dense_filters=growth_rate

    X11=input_x
    dilation_rate=1
    with tf.variable_scope("Network_RDL"):
        for i in range(no_level):
            with tf.variable_scope("Lattice_block"+str(i)):
                for j in range(net_blocks[i]):
                    with tf.variable_scope("micro_block"+str(j)):
                        X11,_=Build_Dense_Lattice_UpDown_v3(X11,dilation_rate,is_training,seq_len,max_drop,no_layers_level[i],
                                      dense_filters[i],block_id=i,num_outputs=num_outputs,net_height=net_height[i])
                        
    X11=tf.layers.dense(tf.boolean_mask(X11, tf.sequence_mask(seq_len)),num_outputs)                    
    return X11,seq_len

def Build_Dense_Lattice_UpDown_v3(X11,dilation_rate,is_training,seq_len,max_drop,no_layers,
                  dense_filters,block_id,num_outputs,net_height=3):
    if (no_layers%2)==0:
        no_layers=no_layers+1
    first_half_layers=(no_layers)//2
    
    first_half_layers=first_half_layers+1
    X_level=[]
    Y_level=[]
    
    ########### first half upward direction
    for l in range(first_half_layers):
        X=[]
        Y=[]
        kernel_stride=1         
        for i in range(np.minimum(l+1,net_height)):
            if l==0:
                X_join=X11
                new_scale_shape=np.shape(X_join)
            elif i==l and l>0:
                X_join=X[i-1]
                new_scale_shape=np.shape(X_join)
            else:
                new_scale_shape=np.shape(Y_level[l-1][i])
                if i>0:
                    X_join=concat_x1x2(Y_level[l-1][i],X[i-1])#
                else:
                    X_join=Y_level[l-1][i]
           
      
                       
                        
            if (l-i)%2==0:
                kernel_size=1
            else:
                kernel_size=2*i+1

            
            dilation_rate=int(np.power(2,i))

            Y_li=conv2d_relu_norm(X_join,kernel_stride,dense_filters[i],
                               is_training,seq_len,max_drop,name="Y"+str(l)+str(i),dilation_rate=dilation_rate,
                               kernel_size=kernel_size)
            if (l-i)>=1:
                new_scale_shape=np.shape(Y_li)
                Y_li=extract_pre_residual(X_level,Y_li,l,i,new_scale_shape,
                                          name="h_skip_block"+str(block_id)+str(l)+str(i),
                                          delta=1,is_training=True,xy="x")
            X.append(X_join)
            Y.append(Y_li)

        X_level.append(X)
        Y_level.append(Y)
                
    ################################ second half: downward direction
    for l in range(first_half_layers,no_layers,+1):
        dilation_rate=1
        X=[]
        Y=[]
        X_revers=[]
        kernel_stride=1
        if l==no_layers-1:
            new_scale_shape=np.shape(Y_level[l-1][0])
            if len(Y_level[l-1])>1:
                X_in=concat_x1x2(Y_level[l-1][0],Y_level[l-1][1])
            else:
                X_in=Y_level[l-1][0]
            ######################################### 
            if (l)%2==0:
                kernel_size=1
            else:
                kernel_size=3
               
            final_filters=dense_filters[0]
            Yf=conv2d_relu_norm(X_in,kernel_stride,final_filters,
                                   is_training,seq_len,max_drop,name="Yf"+str(l),dilation_rate=dilation_rate,
                                   kernel_size=kernel_size)

            if l>1:
                new_scale_shape=np.shape(Yf)
                Yf=extract_pre_residual(X_level,Yf,l,i,new_scale_shape,
                                              name="h_skip_block"+str(block_id)+str(l)+str(i),
                                              delta=1,is_training=True,xy="x")


            X_revers.append(X_in)
            new_scale_shape=np.shape(Yf)
            X11=x_scale(X11,newshape=new_scale_shape,name="Final_block_"+str(i),keep_ch=True)
            Yf=concat_x1x2(Yf,X11)
            
            Y.append(Yf)
        else:
            no_conv_layer_orig=no_layers-l
            no_conv_layer=np.minimum(no_layers-l,net_height)
            for i in range(no_conv_layer-1,-1,-1):
                new_scale_shape=np.shape(Y_level[l-1][i])
                if i==no_conv_layer_orig-1:                   
                    if no_conv_layer_orig<net_height:
                        X_join=concat_x1x2(Y_level[l-1][no_conv_layer_orig],Y_level[l-1][i]) 
                    else:
                        X_join=Y_level[l-1][i]
                       
                else:
                    if i<net_height-1:
                        X_join=concat_x1x2(Y_level[l-1][i],X[no_conv_layer-i-2])
                    else:
                        X_join=Y_level[l-1][i]

                X.append(X_join)
               
            for i in range(no_conv_layer):
                X_in=X[no_conv_layer-i-1]                       
             
                   #############################################################
                if (l-i)%2==0:
                    kernel_size=1
                else:
                    kernel_size=2*i+1
                    
                Y_li=conv2d_relu_norm(X_in,kernel_stride,dense_filters[i],
                                    is_training,seq_len,max_drop,name="Y"+str(l)+str(i),
                                    dilation_rate=dilation_rate,kernel_size=kernel_size)                
                if (l-i)>=1:
                    new_scale_shape=np.shape(Y_li)
                    Y_li=extract_pre_residual(X_level,Y_li,l,i,new_scale_shape,
                                              name="h_skip_block"+str(block_id)+str(l)+str(i),
                                              delta=1,is_training=True,xy="x")
                X_revers.append(X_in)
                Y.append(Y_li)

        X_level.append(X_revers)
        Y_level.append(Y)

    f_y=Y_level[no_layers-1][0]
    
    XY_level=[]
    XY_level.append(X_level)
    XY_level.append(Y_level)
    return f_y,XY_level

def extract_pre_residual(XY_level,X,layer,ix,new_scale_shape,name,delta=2,is_training=True,xy="y",in_2d=False):
    
    pre_res_dens=[X]
    res_ix=layer-delta
    pre_xy1=x_scale(XY_level[res_ix][ix],newshape=new_scale_shape,name=name+str(ix),keep_ch=False,in_2d=in_2d)
    pre_res_dens.append(pre_xy1)
    X=tf.add_n(pre_res_dens, name=name)

    return X
