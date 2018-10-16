import numpy as np
import os
import sys
import uff

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.DEBUG)

x_name = 'x'
y_name = 'output'
input_shape = [None, 480, 480, 1]
frozen_filename = "graph_frozen.pb"
saved_model_dir = "saved/"

def serving_input_receiver_fn():
    inputs = {x_name: tf.placeholder(shape=input_shape, dtype=tf.float32)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def trained_est_to_saved_model(estimator):
    export_dir = estimator.export_savedmodel(
        export_dir_base=saved_model_dir,
        serving_input_receiver_fn=serving_input_receiver_fn)
    return export_dir

def ckpt_to_saved_model(ckpt_path):
    from custom_model_lib import cnn_model_fn
    est_config = tf.estimator.RunConfig()
    drive_classifier = tf.estimator.Estimator(
        config = est_config, model_fn=cnn_model_fn) 
    export_dir = drive_classifier.export_savedmodel(
        export_dir_base=saved_model_dir,
        serving_input_receiver_fn=serving_input_receiver_fn,
        checkpoint_path=ckpt_path)
    print("SavedModel location: " + str(export_dir))
    return export_dir

if __name__ == '__main__':
  ckpt_to_saved_model(sys.argv[1])

