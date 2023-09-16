# -*-coding:utf-8-*-
# @author: Yiqin Qiu
# @email: barryxxz6@gmail.com
# @paper: Steganalysis of adaptive multi-rate speech streams
#         with distributed representations of codewords

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import time
import argparse
from joblib import Parallel, delayed
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from sklearn.metrics import recall_score, accuracy_score
from tensorflow.keras.layers import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_file_list(file_dir, train_num, val_num, test_num):
    train_list = []
    val_list = []
    test_list = []
    for folder in file_dir:
        file_list = []
        for file in os.listdir(folder['folder']):
            file_list.append((os.path.join(folder['folder'], file), folder['class']))
        random.shuffle(file_list)
        train_list += file_list[: train_num // 4]
        val_list += file_list[train_num // 4: (train_num + val_num) // 4]
        test_list += file_list[-test_num // 4:]
    return train_list, val_list, test_list


def read_files(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    array = []
    for line in lines:
        row = [float(item) for item in line.split(' ')]
        array.append(row[0:5])
    return array


class GatedAttention(layers.Layer):

    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 attention_regularizer_weight=0.0,
                 **kwargs):
        super(GatedAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.attention_activation = keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = keras.backend.backend()

        if attention_type == GatedAttention.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == GatedAttention.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

    def get_config(self):
        config = {
            'units': self.units,
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'history_only': self.history_only,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'attention_activation': keras.activations.serialize(self.attention_activation),
            'attention_regularizer_weight': self.attention_regularizer_weight,
        }
        base_config = super(GatedAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.attention_type == GatedAttention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(input_shape)
        elif self.attention_type == GatedAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(input_shape)
        super(GatedAttention, self).build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wa = self.add_weight(shape=(feature_dim, feature_dim),
                                  name='{}_Mul_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Mul_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def call(self, inputs, mask=None, **kwargs):
        input_len = K.shape(inputs)[1]

        if self.attention_type == GatedAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == GatedAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        if self.attention_width is not None:
            if self.history_only:
                lower = K.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower = K.arange(0, input_len) - self.attention_width // 2
            lower = K.expand_dims(lower, axis=-1)
            upper = lower + self.attention_width
            indices = K.expand_dims(K.arange(0, input_len), axis=0)
            e -= 10000.0 * (1.0 - K.cast(lower <= indices, K.floatx()) * K.cast(indices < upper, K.floatx()))
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            e -= 10000.0 * ((1.0 - mask) * (1.0 - K.permute_dimensions(mask, (0, 2, 1))))

        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        a = e / K.sum(e, axis=-1, keepdims=True)

        v = K.batch_dot(a, inputs)
        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        if self.use_attention_bias:
            e = K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = K.reshape(K.dot(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        e = K.batch_dot(K.dot(inputs, self.Wa), K.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e += self.ba[0]
        return e

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_regularizer(self, attention):
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        indices = K.expand_dims(K.arange(0, input_len), axis=0)
        diagonal = K.expand_dims(K.arange(0, input_len), axis=-1)
        eye = K.cast(K.equal(indices, diagonal), K.floatx())
        return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
            attention,
            K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    @staticmethod
    def get_custom_objects():
        return {'GatedAttention': GatedAttention}


def create_classifier(input_shape):
    original_x = Input(shape=input_shape)

    x = Embedding(512, 64)(original_x)
    x = TimeDistributed(Flatten())(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x, y = GatedAttention(attention_activation="relu", return_attention=True)(x)
    x = GlobalAveragePooling1D()(x)

    x = Dense(64, activation="relu")(x)
    logit = Dense(1, activation='sigmoid')(x)

    return Model(inputs=original_x, outputs=logit)


def plot_train_history(history, train_metrics, val_metrics):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = '12'
    plt.plot(history.history.get(train_metrics), '.-', clip_on=False, color='#E50501', label='Train')
    plt.plot(history.history.get(val_metrics), '.-', clip_on=False, color='#230adc', label='Validation')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend()
    ax = plt.gca()
    bwith = 1
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.savefig('no_{}.svg'.format(train_metrics), dpi=1000, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DRCM')
    parser.add_argument('--domain', type=str, default='LPC')
    parser.add_argument('--method', type=str, default='CNV')
    parser.add_argument('--length', type=str, default='1.0')
    parser.add_argument('--em_rate', type=str, default='10')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_num', type=int, default=72000 // 3)
    parser.add_argument('--val_num', type=int, default=12000 // 3)
    parser.add_argument('--test_num', type=int, default=12000 // 3)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    FOLDERS = [
        {"class": 0,
         "folder": "/home/barryxxz/audiodata/AMR_NB/{}/{}/Universe/TXT/Chinese/{}s/0".format(args.domain, args.method, args.length)},
        {"class": 0,
         "folder": "/home/barryxxz/audiodata/AMR_NB/{}/{}/Universe/TXT/English/{}s/0".format(args.domain, args.method, args.length)},
        {"class": 1,
         "folder": "/home/barryxxz/audiodata/AMR_NB/{}/{}/Universe/TXT/Chinese/{}s/{}".format(args.domain, args.method, args.length, args.em_rate)},
        {"class": 1,
         "folder": "/home/barryxxz/audiodata/AMR_NB/{}/{}/Universe/TXT/English/{}s/{}".format(args.domain, args.method, args.length, args.em_rate)}
    ]
    RE_FOLDERS = [
        {"class": 0,
         "folder": "/home/barryxxz/audiodata/AMR_NB/{}/{}/Universe/ReCompress/TXT/Chinese/{}s/0".format(args.domain, args.method, args.length)},
        {"class": 0,
         "folder": "/home/barryxxz/audiodata/AMR_NB/{}/{}/Universe/ReCompress/TXT/English/{}s/0".format(args.domain, args.method, args.length)},
        {"class": 1,
         "folder": "/home/barryxxz/audiodata/AMR_NB/{}/{}/Universe/ReCompress/TXT/Chinese/{}s/{}".format(args.domain, args.method, args.length, args.em_rate)},
        {"class": 1,
         "folder": "/home/barryxxz/audiodata/AMR_NB/{}/{}/Universe/ReCompress/TXT/English/{}s/{}".format(args.domain, args.method, args.length, args.em_rate)}
    ]
    print(FOLDERS)

    model_path = './weights/lpc_weights_{}_{}s_{}.h5'.format(args.method, args.length, args.em_rate)
    print(model_path)

    train_file, val_file, test_file = get_file_list(FOLDERS, args.train_num, args.val_num, args.test_num)
    # re_train_file, re_val_file, re_test_file = get_file_list(RE_FOLDERS, args.train_num, args.val_num, args.test_num)

    x_train = np.array(Parallel(n_jobs=6)(delayed(read_files)(item[0]) for item in train_file))
    # re_x_train = np.array(Parallel(n_jobs=6)(delayed(read_files)(item[0]) for item in re_train_file))
    y_train_ori = np.array([item[1] for item in train_file])

    x_val = np.array(Parallel(n_jobs=6)(delayed(read_files)(item[0]) for item in val_file))
    # re_x_val = np.array(Parallel(n_jobs=6)(delayed(read_files)(item[0]) for item in re_val_file))
    y_val_ori = np.array([item[1] for item in val_file])

    x_test = np.array(Parallel(n_jobs=6)(delayed(read_files)(item[0]) for item in test_file))
    # re_x_test = np.array(Parallel(n_jobs=6)(delayed(read_files)(item[0]) for item in re_test_file))
    y_test_ori = np.array([item[1] for item in test_file])

    print("train num: %d" % len(x_train))
    print("val num: %d" % len(x_val))
    print("test num: %d" % len(x_test))

    tmp = 0
    for item in y_train_ori:
        if item == 0:
            tmp += 1
    print("ratio in train: %.2f" % (tmp / (len(y_train_ori) - tmp)))
    tmp = 0
    for item in y_val_ori:
        if item == 0:
            tmp += 1
    print("ratio in val: %.2f" % (tmp / (len(y_val_ori) - tmp)))
    tmp = 0
    for item in y_test_ori:
        if item == 0:
            tmp += 1
    print("ratio in test: %.2f" % (tmp / (len(y_test_ori) - tmp)))

    in_shape = x_train.shape[1:]

    model = create_classifier(in_shape)
    model.summary()

    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=0, save_best_only=True,
                                 mode='max', save_weights_only=True)
    acc_print_callback = LambdaCallback(
        on_epoch_end=lambda batch, logs:
        print('acc on test set: %.4f' %
              accuracy_score(y_test_ori, model.predict(x_test, batch_size=args.batch_size))))
    fpr_print_callback = LambdaCallback(
        on_epoch_end=lambda batch, logs:
        print('fpr on test set: %.4f' %
              (1 - recall_score(y_test_ori, model.predict(x_test, batch_size=args.batch_size),
                                pos_label=0))))
    fnr_print_callback = LambdaCallback(
        on_epoch_end=lambda batch, logs:
        print('fnr on test set: %.4f' %
              (1 - recall_score(y_test_ori, model.predict(x_test, batch_size=args.batch_size)))))

    callbacks_list = [checkpoint]

    optimizer = Adam(learning_rate=1e-3)
    loss = BinaryCrossentropy()
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    hist = model.fit(x_train, y_train_ori, batch_size=args.batch_size, epochs=args.epoch,
                     validation_data=(x_val, y_val_ori), callbacks=callbacks_list, verbose=2)

    # plot_train_history(hist, 'loss', 'val_loss')
    # plot_train_history(hist, 'accuracy', 'val_accuracy')

    net = create_classifier(in_shape)
    net.load_weights(model_path)
    y_predict = net.predict(x_test)

    y_predict = (y_predict > 0.5)
    # y_predict = np.argmax(y_predict, axis=1)
    print('* accuracy on test set: %0.2f%%' % (accuracy_score(y_test_ori, y_predict) * 100))

    tpr = recall_score(y_test_ori, y_predict)
    tnr = recall_score(y_test_ori, y_predict, pos_label=0)
    fpr = 1 - tnr
    fnr = 1 - tpr
    print('* FPR on test set: %0.2f' % (fpr * 100))
    print('* FNR on test set: %0.2f' % (fnr * 100))

    f = open("./results/result_lpc_log.txt", 'a')
    f.writelines(["\n" + model_path +
                  " Accuracy %0.2f  " % (accuracy_score(y_test_ori, y_predict) * 100) +
                  "FPR %0.2f  " % (fpr * 100) + "FNR %0.2f  " % (fnr * 100)])
    f.close()
