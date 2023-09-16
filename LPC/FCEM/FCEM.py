import os
import random
import time
import keras.backend as K
from keras.layers import Layer
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import recall_score
from keras.layers import Input, Flatten, Dense, MultiHeadAttention
from keras.layers import TimeDistributed, Dropout, Embedding
from keras.models import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

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

model_path = "./model/weights_fcem_100_DN.h5"
seed = 777
epochs = 30
batch_size = 256


class PositionEncoding(Layer):

    def __init__(self, model_dim, **kwargs):
        self._model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        seq_length = inputs.shape[1]
        position_encodings = np.zeros((seq_length, self._model_dim))
        for pos in range(seq_length):
            for i in range(self._model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i - i % 2) / self._model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2])   # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2])   # 2i+1
        position_encodings = K.cast(position_encodings, 'float32')
        return position_encodings

    def compute_output_shape(self, input_shape):
        return input_shape


class Add(Layer):

    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def call(self, inputs):
        input_a, input_b = inputs
        return input_a + input_b

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def get_file_list(folder):
    file_list = []
    for file in os.listdir(folder):
        file_list.append(os.path.join(folder, file))
    return file_list


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


def create_net(input_shape):
    input_fea = Input(shape=input_shape)
    x = Embedding(512, 100)(input_fea)
    x = TimeDistributed(Flatten())(x)
    y = PositionEncoding(500)(x)
    x = Add()([x, y])
    x = MultiHeadAttention(num_heads=8, key_dim=32)(x, x, x)
    x = Flatten()(x)
    x = Dropout(0.6)(x)
    x = Dense(1, activation="sigmoid")(x)
    return Model(input_fea, x)


if __name__ == "__main__":
    random.seed(seed)

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

    in_shape = x_train.shape[1:]

    model = create_net(in_shape)
    model.summary()

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True,
                                 mode='min', save_weights_only=True)
    callbacks_list = [checkpoint]

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
              callbacks=callbacks_list)

    net = create_net(in_shape)
    net.load_weights(model_path)
    start = time.time()
    y_predict = net.predict(x_test)
    end = time.time()
    print(end - start)
    print('* accuracy on test set: %0.2f%%' % (compute_accuracy(y_test, y_predict) * 100))

    # fpr, tpr, thresholds_keras = roc_curve(y_test, y_predict)

    y_predict = (y_predict > 0.5)
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
