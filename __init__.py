#!/usr/bin/env python3.6
import tensorflow as tf
import numpy
from model import *
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

####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################


parameter_dict = {"classlist" : ("benign", "cancer"),
				"DEBUG_MODE" : True,
				"debug_steps" : 10,
				"max_flag" : True,
				"aux_file_name" : "model_aux_files",
				"channel_to_use" : 3, # 0, 1, 2 for individual, 3 for all
				"color_flag": False, #only important if not using all channels
				"quad_img_size" : 256,
				"fold_list" : [0], #### must be list[3,0,9] for all folds use list(range(10))
				"num_folds" : 10,
				"batch_size" : 10,
				"img_size" : 128,
				"adj_img_size" : 125,
				"train_file_in" : "train",
				"test_file_in" : "test",
				"model_file" : "model_stack",
				"train_flag" : True,
				"test_flag" : True,
				"verbose_flag" : True,
				"num_epochs" : 10,
				"write_flag" : True,
				"buffer_size" : 200,
				"early_stop_loss" : 0.05, # set this to zero to disable early stopping
				"write_file" : "MARGResultBook.xlsx",
				"comment" : "first epoch using stacked",
				"img_folder" : "COMBINEDstacked_tiny",
				"age_file_name" : "AAA_PATIENT_AGES_COMINED.txt",
				"path_to_folder" : "/data/", #MAKE SURE TO END WITH /!!!
				"img_string_file" : "efiles.txt",
				"window" : 10,
				"report_interval" : 100,
				"brightness_control" : 0.10, #value between (0,1]
				"HSV_control" : 0.01 #value between (0,1]
				 }
parameter_dict["num_classes"] = len(parameter_dict["classlist"])
parameter_dict["pickle_filename"] = parameter_dict["aux_file_name"] + "/" + "folds.p"


network_dict = {"learning_rate" : 0.0001,
				"num_filts" : [16, 32, 32, 64, 64, 128, 128, 256], # needs to have 8 for small 10 for big
				"fc_neurons" : [1024, 1024, 512], # [2048, 2048, 1024] can be any length
				"dropout" : 0.5,
				"adj_img_size": parameter_dict["adj_img_size"],
				"num_classes" : parameter_dict["num_classes"],
				"print_shape_4_debug_flag" : parameter_dict["DEBUG_MODE"],
				"use_age_flag" : True,
				"batch_norm_flag" : False,
				"fused_batch" : False,
				"batch_norm_decay" : 0.95,
				"act_func" : tf.nn.selu, ## "selu" or tf.nn.elu or tf.nn.relu
				"kernel_size" : 3,
				"view_list" : ["mlo_lil", "mlo_big", "cc_lil", "cc_big"],
				"conv_layer_name" : "Convolution_Layer",
				"pool_layer_name" : "Max_Pool",
				"full_con_name" : "Fully_Connected",
				"dropout_name" : "Dropout",
				"conv_net_name" : "Overall_Conv_Net",
				"num_channel" : 1 if (parameter_dict["channel_to_use"] < 3 and not parameter_dict["color_flag"]) else 3
				}
######################################################

if __name__ == "__main__":

	if parameter_dict["verbose_flag"]:
		tf.logging.set_verbosity(tf.logging.INFO)

	data = Dataset(parameter_dict) # FOR ACTUAL RUNS
	fold_hand = fold_manage(parameter_dict)
	start = fold_hand.fload()
	end = len(parameter_dict["fold_list"])
	for foi in range(start,end):
		######################################################

		train_file_actual, test_file_actual, parameter_dict["train_size"] = data.mayb_write2file(foi)
		parameter_dict["test_file_str"], parameter_dict["test_file_lab"] = data.get_test_classes_and_imgnames()
		parameter_dict["train_file_str"], parameter_dict["train_file_lab"] = data.get_train_classes_and_imgnames()
		parameter_dict["max_step"] = parameter_dict["num_epochs"] * (parameter_dict["train_size"] // parameter_dict["batch_size"] + 1)

		######################################################
		train_pipe = Pipeline(train_file_actual, parameter_dict)
		test_pipe = Pipeline(test_file_actual, parameter_dict, test_flag = True)

		######################################################
		CNN = Network(network_dict)
		par_model = {"num_classes" : parameter_dict["num_classes"],
					"learning_rate" : network_dict["learning_rate"],
					"dropout" : network_dict["dropout"],
					"network2use" : CNN}

		######################################################
		mod_dir = parameter_dict["aux_file_name"] + '/' + parameter_dict["model_file"] + '_' + str(foi) + '.est'
		config = tf.estimator.RunConfig()
		config = config.replace(tf_random_seed=random.randint(0,1e7))
		model = tf.estimator.Estimator(model_fn, model_dir=mod_dir, config=config, params= par_model)
		ehook = EstopHook(parameter_dict)
		tensors_to_log = {'train_accuracy': 'train_accuracy'}
		logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter= parameter_dict["report_interval"])
		hook_list = [ehook, logging_hook]
		#hook_list = [ehook]
		if parameter_dict["DEBUG_MODE"]:
			model.train(input_fn= train_pipe.input_fn, hooks=hook_list, steps=parameter_dict["debug_steps"])
		elif parameter_dict["max_flag"]:
			model.train(input_fn= train_pipe.input_fn, hooks=hook_list, max_steps=parameter_dict["max_step"])
		else:
			model.train(input_fn= train_pipe.input_fn, hooks=hook_list)

		######################################################
		e = model.evaluate(test_pipe.input_fn)
		print('Testing Accuracy:', e['accuracy'])
		predictions = model.predict(lambda : test_pipe.input_fn(True))

		# EXCEL LOGGING #
		Result = Result_Book(parameter_dict,foi)
		Result.log_run(predictions,e)

		######################################################
		fold_hand.fsave(foi)
		###END