import numpy as np
import os
import sys
import uff

import tensorflow as tf
from tensorflow.python.platform import gfile

from custom_model_lib import cnn_model_fn

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
    est_config = tf.estimator.RunConfig()
    drive_classifier = tf.estimator.Estimator(
        config = est_config, model_fn=cnn_model_fn) 
    export_dir = drive_classifier.export_savedmodel(
        export_dir_base=saved_model_dir,
        serving_input_receiver_fn=serving_input_receiver_fn,
        checkpoint_path=ckpt_path)
    print("SavedModel location: " + str(export_dir))
    return export_dir

def create_frozen_graph_from_saved(export_dir):
    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    with tf.Session(graph=tf.Graph()) as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, ["serve"], (export_dir))
        signature = meta_graph_def.signature_def
        x_tensor_name = signature[signature_key].inputs[x_name].name
        y_tensor_name = signature[signature_key].outputs[y_name].name
        x_tensor_list = [x_tensor_name[:-2]]
        y_tensor_list = [y_tensor_name[:-2]]
        graph = tf.get_default_graph()
        graph_in_def = graph.as_graph_def()
        graph_const = tf.graph_util.convert_variables_to_constants(sess, graph_in_def, [y_tensor_name[:-2]]) 
        graph_no_training = tf.graph_util.remove_training_nodes(graph_const)
        with gfile.FastGFile(frozen_filename,'wb') as f:
            f.write(graph_no_training.SerializeToString())
        return graph_no_training

if __name__ == '__main__':
  export_dir = ckpt_to_saved_model(sys.argv[1])
  create_frozen_graph_from_saved(export_dir)
  uff_model = uff.from_tensorflow_frozen_model(frozen_filename, y_tensor_list, output_filename='estimator.uff', text=True)
