import numpy as np
import os
import sys
import uff

import tensorflow as tf
from tensorflow.python.platform import gfile

from custom_model_lib import cnn_model_fn
from estimator_to_saved import ckpt_to_saved_model
from saved_to_frozen import create_frozen_graph_from_saved
tf.logging.set_verbosity(tf.logging.DEBUG)

if __name__ == '__main__':
  export_dir = ckpt_to_saved_model(sys.argv[1])
  create_frozen_graph_from_saved(export_dir)

