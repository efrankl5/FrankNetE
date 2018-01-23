import tensorflow as tf
import numpy
import os
import random
from tensorflow.python.ops import control_flow_ops
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils

########################################################
########################################################
class Network:
	
	def __init__(self, n_dict):
		self.num_filts = n_dict["num_filts"]
		self.fc_neurons = n_dict["fc_neurons"]
		self.num_fc_layers = len(self.fc_neurons)
		self.num_classes = n_dict["num_classes"]
		self.adj_img_size = n_dict["adj_img_size"]
		self.kern = n_dict["kernel_size"]
		self.view_list = n_dict["view_list"]
		self.dropout = n_dict["dropout"]
		self.conv_net_name = n_dict["conv_net_name"]
		self.conv_layer_name = n_dict["conv_layer_name"]
		self.pool_layer_name =  n_dict["pool_layer_name"]
		self.batch_norm_decay = n_dict["batch_norm_decay"]
		self.dropout_name = n_dict["dropout_name"]
		self.full_con_name = n_dict["full_con_name"]
		self.batch_norm_flag = n_dict["batch_norm_flag"]
		self.print_shape_4_debug_flag = n_dict["print_shape_4_debug_flag"]
		self.use_age_flag = n_dict["use_age_flag"]
		self.act_func = n_dict["act_func"]
		self.num_channel = n_dict["num_channel"]
		self.use_fuse = n_dict["fused_batch"]
		self.config_init()
		
		return None
	
	def conv_net(self, feat, reuse, is_training):
		
		feat0 = feat[self.view_list[0]]
		feat1 = feat[self.view_list[1]]
		feat2 = feat[self.view_list[2]]
		feat3 = feat[self.view_list[3]]
		age = tf.reshape(tf.cast(feat["age"], tf.float32), [-1, 1])

		
		fc_neurons = self.fc_neurons
		name = self.conv_net_name
		adj_img_size = self.adj_img_size
		n_classes = self.num_classes
		dropout = self.dropout
		
		if self.print_shape_4_debug_flag and is_training:
			print(name)
			print(self.view_list[0], feat0.shape)
			print(self.view_list[1], feat1.shape)
			print(self.view_list[2], feat2.shape)
			print(self.view_list[3], feat3.shape)
			print("age tensor", age.shape)
		
		with tf.variable_scope(name, reuse=reuse):
			
			
			view0 = self.view_conv_net(feat0, reuse, is_training, 0)
			view1 = self.view_conv_net(feat1, reuse, is_training, 1)
			view2 = self.view_conv_net(feat2, reuse, is_training, 2)
			view3 = self.view_conv_net(feat3, reuse, is_training, 3)
		
			if self.use_age_flag:
				fused = tf.concat([view0, view1, view2, view3, age], 1, name='fused')
			else:
				fused = tf.concat([view0, view1, view2, view3], 1, name='fused')
				
			
			
			if self.print_shape_4_debug_flag and is_training:
				print("age tensor", age.shape)
				print(self.view_list[0], view0.shape)
				print(self.view_list[1], view1.shape)
				print(self.view_list[2], view2.shape)
				print(self.view_list[3], view3.shape)
				print("fused", fused.shape)
			
			fc = [None] * (self.num_fc_layers)
			do = [None] * (self.num_fc_layers)
			
			
			for ind in range(self.num_fc_layers):
				
				if ind == 0:
					fc[ind] = self.layer_fully_connected(fused, fc_neurons[ind], reuse, is_training, ind)
					do[ind] = self.layer_dropout(fc[ind], reuse, is_training, ind)
				else:
					fc[ind] = self.layer_fully_connected(do[ind-1], fc_neurons[ind], reuse, is_training, ind)
					do[ind] = self.layer_dropout(fc[ind], reuse, is_training, ind)
				
			out = tf.layers.dense(inputs=do[-1], units=n_classes, name= "Decider" ,reuse=reuse)
			
			if self.print_shape_4_debug_flag and is_training:
					print("softmax", out.shape)
			
			return out
	
	def view_conv_net(self, x_in, reuse, is_training, view_ind):
		
		adj_img_size = self.adj_img_size
		name = self.view_list[view_ind]
		
		if self.print_shape_4_debug_flag and is_training:
			print(name)
		
		with tf.variable_scope(name, reuse=reuse):
			
			x = tf.reshape(x_in, shape=[-1, adj_img_size, adj_img_size, self.num_channel])
			
			# Block 1
			#256x256 250
			block1 = self.block_conv(x, reuse, is_training, 1)

			# Block 2
			#128x128 125
			block2 = self.block_conv(block1, reuse, is_training, 2)
			
			# Block 3
			#64x64 63
			block3 = self.block_conv(block2, reuse, is_training, 3)
			
			# Block 4
			#32x32 32
			block4 = self.block_conv(block3, reuse, is_training, 4)
			
			if adj_img_size == 250:
			
				# Block 5
				#16x16 16
				block5 = self.block_conv(block4, reuse, is_training, 5)
				out_flat = tf.layers.flatten(block5)
			
			else:
				
				out_flat = tf.layers.flatten(block4)
			
			return out_flat
	
	def block_conv(self, x, reuse, is_training, b_ind):
		
		num_filts = self.num_filts
		kern = self.kern
		
		name = "block_" + str(b_ind)
		
		if self.print_shape_4_debug_flag and is_training:
			print(name)
		
		with tf.variable_scope(name, reuse=reuse):
		
			conv1 = self.layer_conv(x, [kern, kern, num_filts[b_ind]], reuse, is_training, 1)
			conv2 = self.layer_conv(conv1, [kern, kern, num_filts[b_ind + 1]], reuse, is_training, 2)
			pool = self.layer_maxpool_2x2(conv2, reuse, is_training)

			return pool
	
	def layer_conv(self, x, conv_filter, reuse, is_training, ind):
		
		name = self.conv_layer_name + "_" + str(ind)
		
		conv = tf.layers.conv2d(inputs=x, filters=conv_filter[2], kernel_size=conv_filter[0:2], 
								padding="same", activation=self.act_func, 
								kernel_initializer = self.init,
								name=name ,reuse=reuse)
		
				
		if self.batch_norm_flag:
			b_norm = tf.contrib.layers.batch_norm(conv, decay=self.batch_norm_decay,
												  updates_collections=tf.GraphKeys.UPDATE_OPS, is_training=is_training,
												  reuse=reuse, fused=self.use_fuse, scope=name, renorm=False, renorm_decay=0.95)
			return b_norm
		else:
			return conv
		
	def layer_maxpool_2x2(self, x, reuse, is_training):
		
		name = self.pool_layer_name
		
		pool = tf.layers.max_pooling2d(x, pool_size=[2,2], strides=2, padding='same', name=name)
		
		if (self.print_shape_4_debug_flag and is_training):
			print(name, pool.shape)
			
		return pool
	
	def layer_fully_connected(self, x, neuron_num, reuse, is_training, l_ind):
		
		name = self.full_con_name + "_" + str(l_ind)
		
		full_con = tf.layers.dense(inputs=x, units=neuron_num, activation=self.act_func,
								   kernel_initializer = self.init,
								   name=name ,reuse=reuse)
		
		if (self.print_shape_4_debug_flag and is_training):
			print(name, full_con.shape)
		

		return full_con

	def layer_dropout(self, x, reuse, is_training, l_ind):

		name = self.dropout_name + "_" + str(l_ind)

		if self.alpha_drop_flag:
			return utils.smart_cond(is_training, lambda: tf.contrib.nn.alpha_dropout(x, self.dropout, noise_shape=None, seed=None, name=name),
				lambda: array_ops.identity(x))
		else:
			return tf.layers.dropout(inputs=x, rate=self.dropout, training=is_training, name=name)

	def config_init(self):
		act = self.act_func
		if act == tf.nn.selu:
			self.init = layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
			self.alpha_drop_flag = True
		else:
			self.init = None
			self.alpha_drop_flag = False
			return None
########################################################
########################################################
########################################################
########################################################
########################################################{ "keys": ["super+shift+r"],  "command": "reindent" }
	


########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
