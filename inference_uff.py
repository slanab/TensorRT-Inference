import os
import time
import sys

import tensorflow as tf
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from tensorflow.python.client import device_lib

import tensorrt as trt
from tensorrt.parsers import uffparser

from unetp_lib import cnn_model_fn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ckpt_num = 3771463
ckpt_num_folder = 'Checkpoint = ' + str(ckpt_num) + '/'

img_no = '03'
img_name = img_no + "_test"
img_path =   'Images/' + img_name + ".tif"

network_dir = 'Network/'
save_dir = 'output/'

separation = 8
tile_size = 28      # size of the tiles that get input into the NN
threshold = 0.55        # acceptable probability the tile is a vessel

input_pipe_l = []

tf.logging.set_verbosity(tf.logging.ERROR)
print(tf.VERSION)
print device_lib.list_local_devices()
print(tf.VERSION)

if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except OSError:
        pass        

def make_image(input_data, output_name):
    sizes = np.shape(input_data)     
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(input_data, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.savefig(output_name, dpi = sizes[0])
    plt.close() 

def main(unused_argv):

    start_time = time.time()

    # Create the Estimator
    est_config = tf.estimator.RunConfig()
    est_config = est_config.replace(
        model_dir = network_dir,
        session_config=tf.ConfigProto(log_device_placement=False))

    drive_classifier = tf.estimator.Estimator(
            config = est_config, model_fn=cnn_model_fn) 

    if True:
        folder = 'temp/'
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except OSError:
                pass

        print(img_path)
        test_image = np.asarray(misc.imread(img_path), dtype=np.float32) 
        pp_image = test_image

        print(pp_image.shape)
        x_d = pp_image.shape[0]-tile_size+1
        y_d = pp_image.shape[1]-tile_size+1

        # Getting a patch for every pixel in the image, unless separation > 1
        if separation > 1:
            i_sep = int(separation)
            x_values = np.arange(0,x_d,i_sep)
            y_values = np.arange(0,y_d,i_sep)
            if (x_d - 1) % i_sep != 0:
                x_values = np.append(x_values, x_d - 1)
            if (y_d - 1) % i_sep != 0:    
                y_values = np.append(y_values, y_d - 1)
        else:
            x_values = np.arange(0,x_d)
            y_values = np.arange(0,y_d)

        #print(x_values, y_values)

        for x in x_values:
            for y in y_values:
                input_pipe_l.append(test_image[x:(x+tile_size),y:(y+tile_size)])
                #print (str(x) + ':' + str(y) + ' ' + str(x+tile_size) + ':' + str(y+tile_size))

        # input_g follows a shape of (num_patches, 28, 28)
        input_g = np.asarray(input_pipe_l, dtype=np.float32)

        print('Input pipeline constructed.')
        print('Input shape: ' + str(input_g.shape))
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": input_g},
            num_epochs=1,
            shuffle=False)

        # Lana's code for using uff model
        print("====================================");
        img1 = np.ascontiguousarray(test_image[216:244,216:244])
        img2 = np.ascontiguousarray(test_image[244:272,216:244])
        img3 = np.ascontiguousarray(test_image[216:244,244:272])
        img4 = np.ascontiguousarray(test_image[244:272,244:272])
        imgs = np.array([img1, img2, img3, img4])
        start_time = time.time()

        img = img1
        im1_uff_model = open('uff_no_reshape.uff','rb').read()
        out_sq = np.empty([28,28])
        im1_G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
        im1_parser = uffparser.create_uff_parser()
        im1_parser.register_input("Reshape", (1,28,28), 0)
        im1_parser.register_output("output_score/output_relu")
        im1_engine = trt.utils.uff_to_trt_engine(im1_G_LOGGER, im1_uff_model, im1_parser, 1, 1 << 20)
        im1_parser.destroy()
        im1_runtime = trt.infer.create_infer_runtime(im1_G_LOGGER)
        im1_context = im1_engine.create_execution_context()
        im1_output = np.empty(28*28*1, dtype = np.float32)
        im1_d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
        im1_d_output = cuda.mem_alloc(1 * im1_output.size * im1_output.dtype.itemsize)
        im1_bindings = [int(im1_d_input), int(im1_d_output)]
        im1_stream = cuda.Stream()
        ###transfer input data to device
        cuda.memcpy_htod_async(im1_d_input, img, im1_stream)
        ###execute model
        im1_context.enqueue(1, im1_bindings, im1_stream.handle, None)
        ###transfer predictions back
        cuda.memcpy_dtoh_async(im1_output, im1_d_output, im1_stream)
        ###syncronize threads
        im1_stream.synchronize()
        out_sq = np.reshape(im1_output, (28,28))
        trt.utils.write_engine_to_file("./tf_mnist.engine", im1_engine.serialize())
        new_engine = trt.utils.load_engine(im1_G_LOGGER, "./tf_mnist.engine")
        im1_context.destroy()
        #im1_engine.destroy()
        new_engine.destroy()
        im1_runtime.destroy()
        out1 = out_sq
        make_image(out1, folder + img_name + "_uff_out1.png")

        img = img2
        im2_runtime = trt.infer.create_infer_runtime(im1_G_LOGGER)
        context2 = im1_engine.create_execution_context()
        output2 = np.empty(28*28*1, dtype = np.float32)
        d_input2 = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
        d_output2 = cuda.mem_alloc(1 * output2.size * output2.dtype.itemsize)
        bindings2 = [int(d_input2), int(d_output2)]
        stream2 = cuda.Stream()
        cuda.memcpy_htod_async(d_input2, img, stream2)
        context2.enqueue(1, bindings2, stream2.handle, None)
        cuda.memcpy_dtoh_async(output2, d_output2, stream2)
        stream2.synchronize()
        out_sq = np.reshape(output2, (28,28))
        new_engine2 = trt.utils.load_engine(im1_G_LOGGER, "./tf_mnist.engine")
        context2.destroy()
        im1_engine.destroy()
        new_engine2.destroy()
        im2_runtime.destroy()
        out2 = out_sq
        make_image(out2, folder + img_name + "_uff_out2.png")


        out3 = out_sq
        out4 = out_sq

        out_top = np.hstack((out1, out2))
        out_btm = np.hstack((out3, out4))
        out_final = np.vstack((out_top, out_btm))
        print(out_final)
        print (out_final.shape)
        make_image(out_final, folder + img_name + "_uff_out.png")

        print("====================================");




    current_time = time.time() - start_time
    print("Done! Time elapsed: %f seconds." % current_time)    
if __name__ == "__main__":
  tf.app.run()
