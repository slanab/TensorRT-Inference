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
        out_sq = np.empty([28,28])
        out_imgs_ch1 = np.empty([28,28,4])
        out_imgs_ch2 = np.empty([28,28,4])
        start_time = time.time()

        ### General inference setup
        uff_model = open('uff_no_reshape.uff','rb').read()
        G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
        parser = uffparser.create_uff_parser()
        parser.register_input("Reshape", (1,28,28), 0)
        parser.register_output("output_score/output_relu")
        engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, 1, 1 << 20)
        parser.destroy()
        runtime = trt.infer.create_infer_runtime(G_LOGGER)

        output = np.empty(28*28*2, dtype = np.float32)

        for i in range(0,4):
            img = imgs[i]
            context = engine.create_execution_context()
            d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
            d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
            bindings = [int(d_input), int(d_output)]
            stream = cuda.Stream()
            cuda.memcpy_htod_async(d_input, img, stream)
            context.enqueue(1, bindings, stream.handle, None)
            cuda.memcpy_dtoh_async(output, d_output, stream)
            stream.synchronize()
            out_ch1 = np.reshape(output[0:28*28], (28,28))
            out_ch2 = np.reshape(output[28*28:28*28*2], (28,28))
            context.destroy()
            out_imgs_ch1[:,:,i] = out_ch1
            out_imgs_ch2[:,:,i] = out_ch2
            #make_image(out_sq, folder + img_name + str(i) + "_uff_out.png")
            

        ### General inference cleanup
        new_engine = trt.utils.load_engine(G_LOGGER, "./tf_mnist.engine")
        engine.destroy()
        new_engine.destroy()
        runtime.destroy()

        out0 = out_imgs_ch1[:,:,0]
        out1 = out_imgs_ch1[:,:,1]
        out2 = out_imgs_ch1[:,:,2]
        out3 = out_imgs_ch1[:,:,3]

        out_top = np.hstack((out0, out1))
        out_btm = np.hstack((out2, out3))
        out_final = np.vstack((out_top, out_btm))

        make_image(out_final, folder + img_name + "_uff_out_ch1.png")

        out0 = out_imgs_ch2[:,:,0]
        out1 = out_imgs_ch2[:,:,1]
        out2 = out_imgs_ch2[:,:,2]
        out3 = out_imgs_ch2[:,:,3]

        out_top = np.hstack((out0, out1))
        out_btm = np.hstack((out2, out3))
        out_final = np.vstack((out_top, out_btm))

        make_image(out_final, folder + img_name + "_uff_out_ch2.png")

        print("====================================");


    current_time = time.time() - start_time
    print("Done! Time elapsed: %f seconds." % current_time)    
if __name__ == "__main__":
  tf.app.run()
