# TensorRT inference for Estimator model
Running TensorRT-accelerated inference on a network built using Estimatorm API

## Step 1: Create SavedModel for the network to create input and output tensors
Function "export_savedmodel" requries a trained estimator model. For either option below, modify input_shape variable and input name in serving_input_receiver_fn() to reflect your network's input.

Option 1:
If you have a trained network with weights saved in checkpoint format (.data, .index, and .meta files) and a python file containing network model function, run "python estimator_to_saved.py path_to_checkpoint"
This script assumes that the custom model function is called cnn_model_fn and is located in custom_model_lib, import your own model function instead.
Option 2:
Import trained_est_to_saved_model function call after estimator training is complete, and pass already trained estimator as only parameter: trained_est_to_saved_model(drive_classifier). In this case, you don't need to explicitly specify checkpoint or model function.

Either option will create a directory named "saved" containing numerically named folder, ex 1539645300. Run "saved_model_cli show --dir saved/1539645300/ --all" and note signature definitions for input and output. In this case, the script output was 
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:


signature_def['serving_default']:<br />
  The given SavedModel SignatureDef contains the following input(s):<br />
    inputs['x'] tensor_info:<br />
        dtype: DT_FLOAT
        shape: (-1, 480, 480, 1)
        name: Placeholder:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 3)
        name: softmax_tensor:0
  Method name is: tensorflow/serving/predict


## Step 2. Create a frozen TensorFlow model
Run "create_frozen_graph_from_saved path_to_saved_model" (ex create_frozen_graph_from_saved saved/1539645300). Rename variables "x_name" and "y_name" with the signature definition names from the previous steps (here 'x' and 'output'). A frozen graph is saved in the current directory.

At this point, the graph can be treated like any other frozen tensorflow graph (https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_tf_python) 

