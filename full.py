import numpy as np
np.random.seed(123) 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Conv2D
from keras.utils import np_utils
from keras.datasets import mnist
	
from matplotlib import pyplot as plt

model_name = 'my_model'
model_h5 = model_name + '.h5'

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(1,28,28), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(32, (3, 3), input_shape=(1,28,28), data_format='channels_first'))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2,2), padding='same'))

model.add(Convolution2D(32, (3, 3), input_shape=(1,28,28), data_format='channels_first'))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2,2), padding='same'))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=1, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0)
model.save(model_h5)

name_in = model.input.name[:-2]
name_out = model.output.name[:-2]
model_h = model.input.shape[2]
model_w = model.input.shape[3]
print('Training model complete\n')

del model

import keras.backend as K
K.set_learning_phase(0)
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
import tensorrt as trt
import uff
import os

out_folder = "temp/"
fn_frozen = out_folder + model_name + '.pb'
fn_uff = model_name + '.uff'

model = load_model(model_h5)
sess = K.get_session()
saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
checkpoint_path = saver.save(sess, out_folder + 'saved_ckpt', global_step=0, latest_filename='checkpoint_state')
graph_io.write_graph(sess.graph, '.', out_folder + 'tmp.pb')
graph = sess.graph.as_graph_def()
freeze_graph.freeze_graph(out_folder + 'tmp.pb', '', False, checkpoint_path, name_out, "save/restore_all", "save/Const:0", fn_frozen, False, "")
tf.train.write_graph(sess.graph_def, '.', out_folder + 'graph_readable.pbtxt') 

uff_model = uff.from_tensorflow_frozen_model(fn_frozen, [name_out], output_filename=fn_uff, text=True)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 20        
    	parser.register_input(name_in, (1, model_h, model_w))
    	parser.register_output(name_out)
        parser.parse(fn_uff, network)
        engine = builder.build_cuda_engine(network)
