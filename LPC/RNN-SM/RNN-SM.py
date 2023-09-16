# -*-coding:utf-8 -*-
"""
Implementation of our full RNN-SM algorithm
-------------
Based on paper:
    RNN-SM: Fast Steganalysis of VoIP Streams Using Recurrent Neural Network
-------------
"""

import os
import pickle
import random
import time
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Activation, LSTM, Flatten
from keras.models import Sequential
from sklearn.metrics import recall_score

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

TRAIN_FOLDERS = [
    {"class": 1, "folder": "D:\\DataSet\\Vstego_Fea\\Train\\DN-QIM\\Chinese\\100"},
    {"class": 1, "folder": "D:\\DataSet\\Vstego_Fea\\Train\\DN-QIM\\English\\100"},
    {"class": 0, "folder": "D:\\DataSet\\Vstego_Fea\\Train\\DN-QIM\\Chinese\\cover"},
    {"class": 0, "folder": "D:\\DataSet\\Vstego_Fea\\Train\\DN-QIM\\English\\cover"}
]
TEST_FOLDERS = [
    {"class": 1, "folder": "D:\\DataSet\\Vstego_Fea\\Test\\DN-QIM\\Chinese\\100"},
    {"class": 1, "folder": "D:\\DataSet\\Vstego_Fea\\Test\\DN-QIM\\English\\100"},
    {"class": 0, "folder": "D:\\DataSet\\Vstego_Fea\\Test\\DN-QIM\\Chinese\\cover"},
    {"class": 0, "folder": "D:\\DataSet\\Vstego_Fea\\Test\\DN-QIM\\English\\cover"}
]

model_path = "./model/weights_rnn_100_DN.h5"  # The path of model weights
SAMPLE_LENGTH = 1000  # The sample length (ms)
BATCH_SIZE = 32  # batch size
ITER = 30  # number of iteration

"""
Get the paths of all files in the folder
"""


def get_file_list(folder):
    file_list = []
    for file in os.listdir(folder):
        file_list.append(os.path.join(folder, file))
    return file_list


"""
Read codeword file
-------------
input
    file_path
        The path to an ASCII file.
        Each line contains features.
        There are (number of frame) lines in total. 
output
    the list of codewords
"""


def parse_sample(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    array = []
    for line in lines:
        row = line.strip("\r\n\t").strip().split(" ")
        row = list(map(float, row))
        array.append(row)
    file.close()
    return array


def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() > 0.5
    return np.mean(pred == y_true)


"""
Full RNN-SM training and testing
"""
if __name__ == '__main__':
    random.seed(777)

    train_files = [(item, folder["class"]) for folder in TRAIN_FOLDERS for item in get_file_list(folder["folder"])]
    random.shuffle(train_files)
    train_samples_x = [(parse_sample(item[0])) for item in train_files]
    train_samples_y = [item[1] for item in train_files]
    np_train_samples_x = np.asarray(train_samples_x)
    np_train_samples_y = np.asarray(train_samples_y)
    file_num = len(train_files)
    sub_file_num = int(file_num / 5)
    x_val = np_train_samples_x[0: sub_file_num]
    y_val = np_train_samples_y[0: sub_file_num]
    x_train = np_train_samples_x[sub_file_num: file_num]
    y_train = np_train_samples_y[sub_file_num: file_num]

    test_files = [(item, folder["class"]) for folder in TEST_FOLDERS for item in get_file_list(folder["folder"])]
    random.shuffle(test_files)
    test_samples_x = [(parse_sample(item[0])) for item in test_files]
    test_samples_y = [item[1] for item in test_files]
    x_test = np.asarray(test_samples_x)
    y_test = np.asarray(test_samples_y)

    print("Building model")
    model = Sequential()
    model.add(LSTM(50, input_length=int(SAMPLE_LENGTH / 20), input_dim=5, return_sequences=True))  # first layer
    model.add(LSTM(50, return_sequences=True))  # second layer
    model.add(Flatten())  # flatten the spatio-temporal matrix
    model.add(Dense(1))  # output layer
    model.add(Activation('sigmoid'))  # activation function
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    print("Training")
    for i in range(ITER):
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, validation_data=(x_val, y_val))
        # model.save('full_model_%d.h5' % (i + 1))
    model.save_weights(model_path)

    # model.load_weights(model_path)
    y_predict = model.predict(x_test)
    print('* accuracy on test set: %0.2f%%' % (compute_accuracy(y_test, y_predict) * 100))
    y_predict = (y_predict > 0.5)
    # print('* precision on test set: %0.4f' % precision_score(y_test, y_predict))
    # print('* recall on test set: %0.4f' % recall_score(y_test, y_predict))
    # print('* f1-score on test set: %0.4f' % f1_score(y_test, y_predict))
    tpr = recall_score(y_test, y_predict)
    tnr = recall_score(y_test, y_predict, pos_label=0)
    fpr = 1 - tnr
    fnr = 1 - tpr
    print('* FPR on test set: %0.2f' % (fpr * 100))
    print('* FNR on test set: %0.2f' % (fnr * 100))
    f = open("result_dn.txt", 'a')
    f.writelines(["\n" + model_path + " Accuracy %0.2f;" % (
                compute_accuracy(y_test, y_predict) * 100) + " FPR %0.2f;" % (fpr * 100) + "FNR %0.2f" % (fnr * 100)])
    f.close()
