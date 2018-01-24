#!/usr/bin/env python3.6
import tensorflow as tf
import numpy
from data_setup import *
from network_contruct import *
from excel_saver import *
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import random
import pickle
from tensorflow.python.ops import control_flow_ops



######################################################
class Pipeline:
	def __init__(self, in_file, p_dict, test_flag = False):

		self.in_file = in_file
		self.test_flag = test_flag
		self.batch_size =  p_dict["batch_size"]
		self.num_epochs = p_dict["num_epochs"]
		self.num_classes = p_dict["num_classes"]
		self.img_size = p_dict["img_size"]
		self.adj_img_size = p_dict["adj_img_size"]
		self.brightness_control = p_dict["brightness_control"]
		self.HSV_control = p_dict["HSV_control"]
		self.buffer_size = p_dict["buffer_size"]
		self.aux_file = p_dict["aux_file_name"]
		self.channel_to_use = p_dict["channel_to_use"]
		self.broad_flag = p_dict["color_flag"]
		self.plot_flag = False

		return None

	def set_plot_flag(self):

		self.plot_flag = True

		return None


	def broadcast_channel(self, image):
		image_mask = image[:,:, self.channel_to_use]
		image_mask = tf.expand_dims(image_mask, 2)
		if self.broad_flag:
			output_image = tf.concat([image_mask, image_mask, image_mask], 2)
			return output_image
		else:
			mage_mask = image[:,:,self.channel_to_use]
			return image_mask
	
	def apply_with_random_selector(self, x, func, num_cases):
		sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
		# Pass the real x only to one of the func calls.
		return control_flow_ops.merge([func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case) for case in range(num_cases)])[0]
	
	def prep_data_augment(self, image):
		adj_img_size = self.adj_img_size
		if (self.channel_to_use < 3 and self.channel_to_use >= 0):
			image = self.broadcast_channel(image)
			image = image = tf.image.random_brightness(image, max_delta= self.brightness_control)
		else:
			image = self.apply_with_random_selector(image, lambda x, ordering: self.distort_color(x, ordering, fast_mode=False), num_cases=4)
		image = tf.contrib.image.rotate(image, tf.random_uniform(shape=[], minval=0, maxval=180, dtype=tf.float32) * numpy.pi / 180, interpolation='BILINEAR')
		image = tf.random_crop(image, [adj_img_size, adj_img_size, 3])
		if not self.plot_flag:
			image = tf.image.per_image_standardization(image)
		return image
	
	def prep_data_test(self, image):
		adj_img_size = self.adj_img_size
		image = tf.image.resize_image_with_crop_or_pad(image, adj_img_size, adj_img_size)
		if (self.channel_to_use < 3 and self.channel_to_use >= 0):
			image = self.broadcast_channel(image)
		if not self.plot_flag:
			image = tf.image.per_image_standardization(image)
		return image
	
	def distort_color(self, image, color_ordering=0, fast_mode=False, scope=None):
		
		brightness_control = self.brightness_control
		HSV_control = self.HSV_control
		
		if fast_mode:
			if color_ordering == 0:
				image = tf.image.random_brightness(image, max_delta= brightness_control)
				image = tf.image.random_saturation(image, lower= 1 - HSV_control, upper= 1 + HSV_control)
			else:
				image = tf.image.random_saturation(image, lower= 1 - HSV_control, upper= 1 + HSV_control)
				image = tf.image.random_brightness(image, max_delta= brightness_control)
		else:
			if color_ordering == 0:
				image = tf.image.random_brightness(image, max_delta= brightness_control)
				image = tf.image.random_saturation(image, lower= 1 - HSV_control, upper= 1 + HSV_control)
				image = tf.image.random_hue(image, max_delta= HSV_control / 2)
				image = tf.image.random_contrast(image, lower= 1 - HSV_control, upper= 1 + HSV_control)
			elif color_ordering == 1:
				image = tf.image.random_saturation(image, lower= 1 - HSV_control, upper= 1 + HSV_control)
				image = tf.image.random_brightness(image, max_delta= brightness_control)
				image = tf.image.random_contrast(image, lower= 1 - HSV_control, upper= 1 + HSV_control)
				image = tf.image.random_hue(image, max_delta= HSV_control / 2)
			elif color_ordering == 2:
				image = tf.image.random_contrast(image, lower= 1 - HSV_control, upper= 1 + HSV_control)
				image = tf.image.random_hue(image, max_delta= HSV_control / 2)
				image = tf.image.random_brightness(image, max_delta= brightness_control)
				image = tf.image.random_saturation(image, lower= 1 - HSV_control, upper= 1 + HSV_control)
			elif color_ordering == 3:
				image = tf.image.random_hue(image, max_delta= HSV_control / 2)
				image = tf.image.random_saturation(image, lower= 1 - HSV_control, upper= 1 + HSV_control)
				image = tf.image.random_contrast(image, lower= 1 - HSV_control, upper= 1 + HSV_control)
				image = tf.image.random_brightness(image, max_delta= brightness_control)
			else:
				raise ValueError('color_ordering must be in [0, 3] but is', color_ordering)
				
		return image
	
	def record_parser(self, record):
		
		test_flag = self.test_flag
		img_size = self.img_size
		num_classes = self.num_classes
		
		# Use `tf.parse_single_example()` to extract data from a `tf.Example`
		# protocol buffer, and perform any additional per-record preprocessing.
		
		if test_flag:
			
			features = {'test/label': tf.FixedLenFeature([], tf.int64),
					   'test/age': tf.FixedLenFeature([], tf.int64),
					   'test/mlo_lil': tf.FixedLenFeature([], tf.string),
					   'test/mlo_big': tf.FixedLenFeature([], tf.string),
					   'test/cc_lil': tf.FixedLenFeature([], tf.string),
					   'test/cc_big': tf.FixedLenFeature([], tf.string)}
			
			parsed = tf.parse_single_example(record, features)
			
			# Perform additional preprocessing on the parsed data.
			#Convert the image data from string back to the numbers
			image_mlo_lil = tf.decode_raw(parsed['test/mlo_lil'], tf.float32)
			image_mlo_big = tf.decode_raw(parsed['test/mlo_big'], tf.float32)
			image_cc_lil = tf.decode_raw(parsed['test/cc_lil'], tf.float32)
			image_cc_big = tf.decode_raw(parsed['test/cc_big'], tf.float32)
			
			# Cast label and age data into int32
			label = tf.cast(parsed['test/label'], tf.int32)
			age = tf.cast(parsed['test/age'], tf.int32)
			
			# Reshape image data into the original shape
			image_mlo_lil = tf.reshape(image_mlo_lil, [img_size, img_size, 3])
			image_mlo_big = tf.reshape(image_mlo_big, [img_size, img_size, 3])
			image_cc_lil = tf.reshape(image_cc_lil, [img_size, img_size, 3])
			image_cc_big = tf.reshape(image_cc_big, [img_size, img_size, 3])
			
			# Any preprocessing here ...
			image_mlo_lil = self.prep_data_test(image_mlo_lil)
			image_mlo_big = self.prep_data_test(image_mlo_big)
			image_cc_lil = self.prep_data_test(image_cc_lil)
			image_cc_big = self.prep_data_test(image_cc_big)
			
			
			label = tf.one_hot(label, num_classes)
			
			return {"mlo_lil" : image_mlo_lil,
				   "mlo_big" : image_mlo_big,
				   "cc_lil" : image_cc_lil,
				   "cc_big" : image_cc_big,
				   "age": age}, label
		
		else:
			
			features = {'train/label': tf.FixedLenFeature([], tf.int64),
					   'train/age': tf.FixedLenFeature([], tf.int64),
					   'train/mlo_lil': tf.FixedLenFeature([], tf.string),
					   'train/mlo_big': tf.FixedLenFeature([], tf.string),
					   'train/cc_lil': tf.FixedLenFeature([], tf.string),
					   'train/cc_big': tf.FixedLenFeature([], tf.string)}
			
			parsed = tf.parse_single_example(record, features)
			
			# Perform additional preprocessing on the parsed data.
			#Convert the image data from string back to the numbers
			image_mlo_lil = tf.decode_raw(parsed['train/mlo_lil'], tf.float32)
			image_mlo_big = tf.decode_raw(parsed['train/mlo_big'], tf.float32)
			image_cc_lil = tf.decode_raw(parsed['train/cc_lil'], tf.float32)
			image_cc_big = tf.decode_raw(parsed['train/cc_big'], tf.float32)
			
			# Cast label and age data into int32
			label = tf.cast(parsed['train/label'], tf.int32)
			age = tf.cast(parsed['train/age'], tf.int32)
			
			# Reshape image data into the original shape
			image_mlo_lil = tf.reshape(image_mlo_lil, [img_size, img_size, 3])
			image_mlo_big = tf.reshape(image_mlo_big, [img_size, img_size, 3])
			image_cc_lil = tf.reshape(image_cc_lil, [img_size, img_size, 3])
			image_cc_big = tf.reshape(image_cc_big, [img_size, img_size, 3])
			
			# Any preprocessing here ...
			image_mlo_lil = self.prep_data_augment(image_mlo_lil)
			image_mlo_big = self.prep_data_augment(image_mlo_big)
			image_cc_lil = self.prep_data_augment(image_cc_lil)
			image_cc_big = self.prep_data_augment(image_cc_big)
			
			
			label = tf.one_hot(label, num_classes)
			
			return {"mlo_lil" : image_mlo_lil,
				   "mlo_big" : image_mlo_big,
				   "cc_lil" : image_cc_lil,
				   "cc_big" : image_cc_big,
				   "age": age}, label
	
	def input_fn(self, predict_flag = False):
		
		test_flag = self.test_flag
		in_file = self.in_file
		num_classes = self.num_classes
		batch_size = self.batch_size
		num_epochs = self.num_epochs
		num_classes = self.num_classes
		b_size = self.buffer_size
		
		filenames = [in_file]
		
		if test_flag:
			if predict_flag:

				dataset_test = tf.data.TFRecordDataset(filenames)

				# Use `Dataset.map()` to build a pair of a feature dictionary and a label 
				# tensor for each example.
				dataset_test = dataset_test.map(map_func=self.record_parser)
				dataset_test = dataset_test.batch(batch_size)
				dataset_test = dataset_test.repeat(1)
				iterator_test = dataset_test.make_one_shot_iterator()

				# `features` is a dictionary in which each value is a batch of values for
				# that feature; `labels` is a batch of labels.
				features, labels = iterator_test.get_next()

				return features

			else:

				dataset_predict = tf.data.TFRecordDataset(filenames)

				# Use `Dataset.map()` to build a pair of a feature dictionary and a label 
				# tensor for each example.
				dataset_predict = dataset_predict.map(map_func=self.record_parser)
				dataset_predict = dataset_predict.batch(batch_size)
				dataset_predict = dataset_predict.repeat(1)
				iterator_predict = dataset_predict.make_one_shot_iterator()

				# `features` is a dictionary in which each value is a batch of values for
				# that feature; `labels` is a batch of labels.
				features, labels = iterator_predict.get_next()

				return features, labels
			
		else:
			
			dataset = tf.data.TFRecordDataset(filenames)

			# Use `Dataset.map()` to build a pair of a feature dictionary and a label 
			# tensor for each example.
			dataset = dataset.map(self.record_parser)
			if not self.plot_flag:
				dataset = dataset.shuffle(buffer_size = b_size, seed = random.randint(0,1e7),  reshuffle_each_iteration=True)
			dataset = dataset.batch(batch_size)
			dataset = dataset.repeat(num_epochs)
			iterator = dataset.make_one_shot_iterator()

			# `features` is a dictionary in which each value is a batch of values for
			# that feature; `labels` is a batch of labels.
			features, labels = iterator.get_next()
			
			return features, labels

######################################################
######################################################
def model_fn(features, labels, mode, params):
	# Build the neural network
	# Because Dropout have different behavior at training and prediction time, we
	# need to create 2 distinct computation graphs that still share the same weights.
	num_classes = params["num_classes"]
	learning_rate = params["learning_rate"]
	dropout = params["dropout"]
	CNN_in = params["network2use"]

	training = mode == tf.estimator.ModeKeys.TRAIN


	logits = CNN_in.conv_net(features, reuse=None, is_training=training)

	with tf.variable_scope('Predictions'):
		predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor"), 
		"logits": logits}

	# If prediction mode, early return
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions=predictions)

	def trainer(losses):
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_out = optimizer.minimize(losses, global_step=tf.train.get_global_step())
			return train_out

	
	
		
	learn_out = tf.train.exponential_decay(learning_rate, global_step=tf.train.get_global_step(), decay_steps=10000, decay_rate=0.96, staircase=True)
	# Define loss and optimizer
	with tf.variable_scope('Optimization'):
		optimizer = tf.train.AdamOptimizer(learning_rate=learn_out)

	loss_op = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, weights=1.0)
	train_op = trainer(loss_op)
		
	if mode == tf.estimator.ModeKeys.TRAIN:
		
		accuracy = tf.metrics.accuracy(labels = tf.argmax(labels, axis=1), predictions = tf.argmax(logits, axis=1))
		tf.identity(accuracy[1], name='train_accuracy')
		tf.summary.scalar('train_accuracy', accuracy[1])

	# Evaluate the accuracy of the model
	with tf.variable_scope('EvaluationOperations'):
		reg_lab = tf.argmax(labels, axis=1)
		acc_op = tf.metrics.accuracy(labels=reg_lab, predictions=predictions["classes"])
	# TF Estimators requires to return a EstimatorSpec, that specify
	# the different ops for training, evaluating, ...
	estim_specs = tf.estimator.EstimatorSpec(
	mode=mode,
	predictions=predictions,
	loss=loss_op,
	train_op=train_op,
	eval_metric_ops={'accuracy': acc_op})

	return estim_specs

######################################################
class EstopHook(tf.train.SessionRunHook):
	def __init__(self, p_dict):
		
		self.estopval = p_dict["early_stop_loss"]
		self.train_size = p_dict["train_size"]
		self.batch_size = p_dict["batch_size"]
		self.loss_array = [99]*p_dict["window"]
		self.report_every = p_dict["report_interval"]
		self.mean = 9999
		
	def begin(self):
		# You can add ops to the graph here.
		print('Early Stop Hook Successfully Added')
		with tf.variable_scope('EarlyStop'):
			# self.compareloss = tf.greater_equal(tf.losses.get_total_loss(), self.estopval, name='EarlyStopCheck')
			self.loss_grab = tf.losses.get_total_loss()
			self.glob_step = tf.train.get_global_step()
	def before_run(self, run_context):
		# return tf.train.SessionRunArgs(self.compareloss)
		return tf.train.SessionRunArgs([self.loss_grab,self.glob_step])
	def after_run(self, run_context, run_values):
		loss_val = run_values.results[0]
		g_step = run_values.results[1]

		old = self.loss_array
		new = old[:-1]
		new.insert(0,loss_val)
		self.loss_array = new
		self.mean = numpy.mean(new)

		epoch = g_step*self.batch_size/self.train_size
		#print check#
		#if g_step % self.report_every == 1:
		#    print("Step/Epoch: %i/%.2f\tCurrent/Average Loss: %.4f/%.4f" % (g_step,epoch,loss_val,self.mean))
		#early stop check#
		if  self.mean < self.estopval:
			run_context.request_stop()
			print('\nENDED THIS SESSION EARLY BECAUSE AVERAGE LOSS WAS %.4f WHICH IS LESS THAN %.4f!\n' % (self.mean,self.estopval))
		return None

######################################################
class fold_manage:
	def __init__(self, p_dict):
		self.fold_list = p_dict["fold_list"]
		p_file = p_dict["pickle_filename"]
		self.p_file = p_file
		if not os.path.isfile(p_file):
			s_ind = 0
			pickle.dump( s_ind, open( p_file, "wb" ) )
			
		return None
	
	def fload(self):
		s_ind = pickle.load( open( self.p_file, "rb" ) )
		
		return s_ind
	
	def fsave(self, c_ind):
		
		pickle.dump( c_ind + 1, open( self.p_file, "wb" ) )
		print("____ DONE WITH FOLD", self.fold_list[c_ind], "____")
		
		return None

######################################################
######################################################
