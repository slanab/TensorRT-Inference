import numpy as np
import os
import sys
import uff

import tensorflow as tf
from tensorflow.python.platform import gfile

tf.logging.set_verbosity(tf.logging.DEBUG)

x_name = 'x'
y_name = 'output'
frozen_filename = "graph_frozen.pb"
saved_model_dir = "saved/"

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
  create_frozen_graph_from_saved(sys.argv[1])

