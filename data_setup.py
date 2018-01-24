#!/usr/bin/env python3
"""
Created on Wed Aug 23 12:06:16 2017

@author: Eli
"""
import os
import re
import numpy as np
import random
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf



class Dataset:
    
    def __init__(self, p_dict):

        self.p_dict = p_dict

        filelisttext_ = p_dict["img_string_file"]
        filepath = p_dict["img_folder"]
        img_size = p_dict["quad_img_size"]
        aux_file = p_dict["aux_file_name"]
        path = p_dict["path_to_folder"]
        """
        INPUT:
            filelisttext-> (str) the file name that has all of the paths to pictures
        """
        # Checking to make sure we got a file by that name

        self.aux_file = aux_file

        try:
            os.makedirs(aux_file)
        except OSError:
            if not os.path.isdir(aux_file):
                raise


        filelisttext = self.gentextfile(path+filepath,filelisttext_)

        assert (os.path.isfile(filelisttext)), 'There is no file by that name'
        
        #reading File into a variable allData
        with open(filelisttext) as file:
        	allData = file.readlines()
        allData = [x.strip() for x in allData]
        
        self.master_file_queue = allData
        self.filepath = filepath + "/"
        self.filepath_noslash = filepath
        self.img_size = img_size

    def gentextfile(self, doi, text_file_name):

        tfn = self.aux_file + '/' + text_file_name

        if os.path.isfile(tfn):
            print('file list found')

        else:
            print('file list not found, generating now')
            the_files = os.listdir(doi)
            with open(tfn, 'w') as file:
                for item in the_files:
                    file.write("{}\n".format(item))

        return tfn
    
    def getMFQ(self):
        return self.master_file_queue
    
    def make_train_and_test(self, shuffle=True, splitter= ".", split_ind=1):

        foi = self.foi
        p_dict = self.p_dict
        classlist = p_dict["classlist"]
        num_folds = p_dict["num_folds"]

        assert (foi <= num_folds), 'The fold you want is to big'
        
        self.num_classes = len(classlist)
        self.num_folds = num_folds
        self.foi = foi
        self.splitter = splitter
        self.split_ind = split_ind
        self.shuffle = shuffle
        
        #splitting data into classes based on classlist
        dataSplit = {}
        ind = 0
        sflag = 0
        allData = self.getMFQ()
        
        for clss in classlist:
        	clssMatch = []
        	for line in allData:
        		sflag = 1
        		for c in clss:
        			search = re.search(c, line)
        			if search is None:
        				sflag = 0
        		if sflag:
        			clssMatch.append(line)
        	dataSplit[ind] = clssMatch
        	ind += 1
            
            
        train = [] 
        test = []
    
    
        for i in dataSplit:
            temp_train, temp_test = self.make_a_class(dataSplit[i], i)
            for tr in temp_train:
                train.append(tr)
            for te in temp_test:
               test.append(te)
        
        
        
        self.train_length = len(train)
        self.test_length = len(test)
        self.train_count = 0
        self.test_count = 0


        if shuffle==True:
            random.shuffle(train)
            
        
        train_file_str = []
        train_file_lab = []
        test_file_str = []
        test_file_lab = []
        
        for items in train:
            train_file_str.append(items[0])
            train_file_lab.append(items[1])
            
        for items in test:
            test_file_str.append(items[0])
            test_file_lab.append(items[1])
        
        self.train_file_str = train_file_str
        self.train_file_lab = train_file_lab
        self.test_file_str = test_file_str
        self.test_file_lab = test_file_lab
        
        print("Done Making Classes")

        for key in dataSplit:
        	print("Class:", key, "\tCriteria:", classlist[key], "\tInstances:", len(dataSplit[key]))

        return None
    
    def make_a_class(self, current_class, label):
        """
        INPUTS:
            current_class-> (list) an unlabeled entire class
            label-> (int) class label, should be unique among other classes
        OUTPUTS:
            current_train-> (list) each element is a list with filepath str and label
            current_test-> (list) each element is a list with filepath str and label
        """
        
        curr_aug_class = self.parser(current_class)
        curr_aug_train,curr_aug_test = self.kfold(curr_aug_class)
        current_test = self.aug_list(curr_aug_test, current_class)
        current_train = self.aug_list(curr_aug_train, current_class)
        
        for i in range(len(current_test)):
            current_test[i] = [current_test[i], label]
            
        for i in range(len(current_train)):
            current_train[i] = [current_train[i], label]
        
        return current_train, current_test
    
    def aug_list(self, plist, nlist):
        """
        INPUTS:
            plist-> (list) a list of patterns
            nlist-> (list) a list of strings wanting to compare for each pattern
        OUTPUTS:
            fin_list-> (list) a list of strings containing at least one pattern
        NOTE:
            patters are not exclusive so if multiple patterns have some overlap then were boned
        """
        fin_list = []
        for p in plist:
            for n in nlist:
                search = re.search(p, n)
                if search is not None:
                    fin_list.append(n)
                
        return fin_list
    
    def parser(self, clss_list):
        """
        INPUT:
            clss_list-> (list) a list of strings
        OUTPUT:
            uniques-> (list) a list that has no repeating strings of interest
        """
        splitter = self.splitter
        split_ind = self.split_ind
        
        uniques = []
        
        for i in range(len(clss_list)):
            coi = clss_list[i].split(splitter)
            if coi[split_ind] not in uniques:
                uniques.append(coi[split_ind])
        
        return uniques
    
    def kfold(self, total_data):
        """
        INPUTS: 
            total_data-> (list) one dimenionsal list of any length
        OUTPUTS:
            train-> (list) one dimenionsal list of training data
            test-> (list) one dimenionsal list of testing data
        """
        num_folds = self.num_folds
        foi = self.foi
        
               
        adj_foi = foi
        total_length = len(total_data)
        fold_dict = {}
        
        ####LETS MAKE A DICTIONARY WTIH ALL SLICES
        for i in range(num_folds):
            fold_dict[i] = []
            
        ####LETS CALCULATE WHERE WE ARE AT WRT EACH FOLD
        ideal_fold = total_length/num_folds
        real_fold = np.floor(total_length/num_folds)
        fold_dif = ideal_fold-real_fold
        
        
        
        #####MAKING THE MAGIC
        total_ind = 0
        carry = 0
        
        for i in range(num_folds):
            carry += fold_dif
            if carry > 1:
                bump = 1
                carry -= 1
            else:
                bump = 0
            if i == (num_folds-1):
                this_fold = total_length - total_ind
            else:
                this_fold = real_fold + bump
            for ii in range(int(this_fold)):
                fold_dict[i].append(total_data[total_ind])
                total_ind += 1
                       
        
        #######SEPARTING INTO TRAIN AND TEST
        test = fold_dict[adj_foi]
        train = []
        for i in fold_dict:
            if i != adj_foi:
                for ii in range(len(fold_dict[i])):
                    train.append(fold_dict[i][ii])
                    
        #######RETURN
        return (train,test)
        
    
    def mayb_write2file(self, foi, shuffle= True):
        


        p_dict = self.p_dict

        self.foi = foi
        self.make_train_and_test(shuffle=shuffle)

        in_train_filename = p_dict["train_file_in"]
        in_test_filename = p_dict["test_file_in"]
        path = p_dict["path_to_folder"]
        age_file = p_dict["age_file_name"]

        

        fold = foi
        folder_name = self.filepath_noslash
        img_size = self.img_size
        self.path = path
        self.age_file = folder_name + '/' + age_file
        train_file_str = self.train_file_str
        train_file_lab = self.train_file_lab
        test_file_str = self.test_file_str
        test_file_lab = self.test_file_lab
        
        train_filename = self.aux_file + '/' + in_train_filename + '.' + folder_name + '.' + str(img_size) + '.' + 'FOLD_' + str(fold) + '.tfrecords'
        test_filename =  self.aux_file + '/' + in_test_filename + '.' + folder_name + '.' + str(img_size) + '.' + 'FOLD_' + str(fold) + '.tfrecords'

        self.read_age_file()
        print("Information about fold", fold)
        print("Train file stats:")
        print("Train filename:",train_filename)
        class_counter = [0]*self.num_classes
        class_ID = list(range(self.num_classes))
        for i in train_file_lab:
            current_ind = class_ID.index(i)
            class_counter[current_ind] += 1
        for i in range(self.num_classes):
            print("Class: ", class_ID[i], "\tInstances: ", class_counter[i])
        print("Test filename:",test_filename)
        class_counter = [0]*self.num_classes
        class_ID = list(range(self.num_classes))
        for i in test_file_lab:
            current_ind = class_ID.index(i)
            class_counter[current_ind] += 1
        for i in range(self.num_classes):
            print("Class: ", class_ID[i], "\tInstances: ", class_counter[i])
        

        file_written_flag = os.path.isfile(train_filename) and os.path.isfile(test_filename)

        if file_written_flag:
            print("Files are already in place, using existing for this run\n")
            return train_filename,test_filename,len(self.train_file_lab)
        
        print("Making Binary TfRecords\nThis will take some time\n")

        

        def _int64_feature(value):
          return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        def _bytes_feature(value):
          return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))       

        writer = tf.python_io.TFRecordWriter(train_filename)
        for i in range(len(train_file_str)):
            # print how many images are saved every 1000 images
            if not i % 100:
                print('Train data: {}/{}'.format(i, len(train_file_str)))
            # Load the image
            mlo_lil, mlo_big, cc_lil, cc_big = self.load_image(train_file_str[i])
            age = self.get_age_for_str(train_file_str[i])
            label = train_file_lab[i]
            # Create a feature
            feature = {'train/label': _int64_feature(label),
                       'train/age': _int64_feature(age),
                       'train/mlo_lil': _bytes_feature(tf.compat.as_bytes(mlo_lil.tostring())),
                       'train/mlo_big': _bytes_feature(tf.compat.as_bytes(mlo_big.tostring())),
                       'train/cc_lil': _bytes_feature(tf.compat.as_bytes(cc_lil.tostring())),
                       'train/cc_big': _bytes_feature(tf.compat.as_bytes(cc_big.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
            
        writer.close()       

        writer = tf.python_io.TFRecordWriter(test_filename)
        for i in range(len(test_file_str)):
            # print how many images are saved every 1000 images
            if not i % 50:
                print('Test data: {}/{}'.format(i, len(test_file_str)))
            # Load the image
            mlo_lil, mlo_big, cc_lil, cc_big = self.load_image(test_file_str[i])
            age = self.get_age_for_str(test_file_str[i])
            label = test_file_lab[i]
            # Create a feature
            feature = {'test/label': _int64_feature(label),
                       'test/age': _int64_feature(age),
                       'test/mlo_lil': _bytes_feature(tf.compat.as_bytes(mlo_lil.tostring())),
                       'test/mlo_big': _bytes_feature(tf.compat.as_bytes(mlo_big.tostring())),
                       'test/cc_lil': _bytes_feature(tf.compat.as_bytes(cc_lil.tostring())),
                       'test/cc_big': _bytes_feature(tf.compat.as_bytes(cc_big.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        writer.close()

        print("All Records written successfully\n")
        return train_filename,test_filename,len(self.train_file_lab)

    def get_test_classes_and_imgnames(self):
        return self.test_file_str, self.test_file_lab
    def get_train_classes_and_imgnames(self):
        return self.train_file_str, self.train_file_lab

    def load_image(self, addr):
        img_size = self.img_size
        
        img_stuff = self.filepath_noslash + '/' +addr
        sub_img_size = self.img_size//2
        
        img_array = mpimg.imread(self.path + self.filepath + addr)

        mlo_lil_out = img_array[:sub_img_size,:sub_img_size,:]
        mlo_big_out = img_array[sub_img_size:,:sub_img_size,:]
        cc_lil_out = img_array[:sub_img_size,sub_img_size:,:]
        cc_big_out = img_array[sub_img_size:,sub_img_size:,:]

        mlo_lil_out = mlo_lil_out[:sub_img_size,:sub_img_size,:]
        mlo_big_out = mlo_big_out[:sub_img_size,:sub_img_size,:]
        cc_lil_out = cc_lil_out[:sub_img_size,:sub_img_size,:]
        cc_big_out = cc_big_out[:sub_img_size,:sub_img_size,:]
        
        return mlo_lil_out, mlo_big_out, cc_lil_out, cc_big_out

    def make_oneHOTS(self, labels):
        
        num_classes = self.num_classes
        labels = np.array(labels)
        num_labels = labels.shape[0]                                                
        index_offset = np.arange(num_labels) * num_classes                       
        ohv = np.zeros((num_labels, num_classes))                                
        ohv.flat[index_offset + labels.ravel()] = 1
        
        return ohv
        
    def reshuffle_train(self):
        
        img_str = self.train_file_str
        labels = self.train_file_lab
        
        assert (len(labels) == len(img_str)), "Lengths Do not match. That is all"
        
        length = len(labels)
        combined = [None]*length
        new_lab = [None]*length
        new_str = [None]*length
        
        for i in range(length):
            combined[i] = [img_str[i],labels[i]]
        random.shuffle(combined)
        for i in range(length):
            new_str[i], new_lab[i] = combined[i]
        
        self.train_file_str = new_str
        self.train_file_lab = new_lab
        
        
        return None
    
    def read_age_file(self):
        age_file = self.p_dict["path_to_folder"] + self.age_file
        age_list = []
        file_ref_list = []
        with open(age_file, "r") as file:
            for line in file:
                line = line.replace("\n","")
                line = line.split("\t")
                age_list.append(line[1])
                file_ref_list.append(line[0])
        self.age_list = age_list
        self.file_ref_list = file_ref_list
        return None
    
    def get_age_for_str(self, in_str):
        age_list = self.age_list 
        file_ref_list = self.file_ref_list
        in_str_split = in_str.split('.')
        ref_str = in_str_split[2]
        found_ind = -1;
        for index, item in enumerate(file_ref_list):
            if item == ref_str:
                found_ind = index
                
        age_out = age_list[found_ind]
        if found_ind == -1:
            print("ERROR: No age found for", in_str, "setting to 60")
            age_out = 60
        return int(age_out)
#####DEBUGGING SECTION#####
if __name__ == "__main__":
    classlist = ("benign", "cancer")
    num_folds = 10
    foi = 3
    batch_size = 10
    img_size = 256
    adj_img_size = 250
    train_file_in = "train"
    test_file_in = "test"
    model_file = "model_stack.est"
    train_flag = True
    test_flag = True
    verbose_flag = True
    num_epochs = 10
    write_flag = True
    learning_rate = 0.0001
    dropout = 0.5 # Dropout, probability to keep units
    early_stop_loss = 0.05 # set this to zero to disable early stopping
    write_file = "MARGResultBook.xlsx"
    comment = "first epoch using stacked"
    img_folder = "COMBINEDstacked_tiny"
    age_file_name = "PATIENT_AGES_COMINED.txt"
#    path_to_folder = "/media/franklondo/Seagate Backup Plus Drive/Eli/" #MAKE SURE TO END WITH /!!!
    img_string_file = "efiles.txt"
    window = 10
    report_interval = 100
    #END DEBUG INPUTS
    	
    gentextfile(img_folder,img_string_file)
#    gentextfile(path_to_folder+'/'+img_folder,img_string_file)
    data = Dataset(img_string_file,img_folder,img_size) # FOR ACTUAL RUNS
    # data = Dataset("filelist.txt","NNDBdev",img_size) # FOR DEBUGGING
    data.make_train_and_test(classlist, num_folds, foi, True)
    num_classes = len(classlist)
    train_file,test_file,train_size = data.mayb_write2file(train_file_in, test_file_in, "", age_file_name)
#    train_file,test_file,train_size = data.mayb_write2file(train_file_in, test_file_in,path_to_folder)
    test_file_str, test_file_lab = data.get_test_classes_and_imgnames()
    train_file_str, train_file_lab = data.get_train_classes_and_imgnames()
    train_info = list(zip(train_file_lab,train_file_str))

    age_list = data.age_list
    file_ref_list = data.file_ref_list

##
    img_stuff = img_folder + '/' +test_file_str[0]
    sub_img_size = img_size//2

    raw_img = Image.open(img_stuff)
    fake_img = mpimg.imread(img_stuff)
    
    img = raw_img.resize((img_size,img_size), Image.ANTIALIAS)
    img_array = np.asarray(img, dtype=np.float32)
    mlo_lil = img_array[:sub_img_size,:sub_img_size,:]
    mlo_big = img_array[sub_img_size:,:sub_img_size,:]
    cc_lil = img_array[:sub_img_size,sub_img_size:,:]
    cc_big = img_array[sub_img_size:,sub_img_size:,:]
#########
    in_str = test_file_str[0]
    age_list = data.age_list 
    file_ref_list = data.file_ref_list
    ##
    in_str_split = in_str.split('.')
    ref_str = in_str_split[2]
    found_ind = -1;
    for index, item in enumerate(file_ref_list):
        if item == ref_str:
            found_ind = index
            
    age_out = age_list[found_ind]