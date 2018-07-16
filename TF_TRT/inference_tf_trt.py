from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
import time
import tensorflow as tf
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile

model = "graph_optimized.pb"

inp_dims = (32, 28, 28, 1)
out_nodes = ["output_score/output_relu"]

input_pipe_l = []

tf.logging.set_verbosity(tf.logging.DEBUG)
print(tf.VERSION)

tile_size = 28      # size of the tiles that get input into the NN
threshold = 0.55        # acceptable probability the tile is a vessel
separation = 8
img_no = '03'       # corresponds to a drive image - KEEP AS STR
img_name = img_no + "_test"
img_path =   '../Images/' + img_name + ".tif"

save_dir = 'output/'

if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except OSError:
        pass        

def make_image(input_data, output_name):
    output_name = save_dir + output_name
    sizes = np.shape(input_data)     
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(input_data, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.savefig(output_name, dpi = sizes[0])
    plt.close() 

test_image = np.asarray(misc.imread(img_path), dtype=np.float32) 
pp_image = test_image   
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

timelineName = None
num_loops=1

def getGraph():
  with gfile.FastGFile(model,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

def timeGraph(gdef, dummy_input):
  tf.logging.info("Starting execution")
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
  tf.reset_default_graph()
  g = tf.Graph()

  outlist=[]
  with g.as_default():
    inc=tf.constant(dummy_input, dtype=tf.float32)
    dataset=tf.data.Dataset.from_tensors(inc)
    dataset=dataset.repeat()
    iterator=dataset.make_one_shot_iterator()
    next_element=iterator.get_next()
    out = tf.import_graph_def(
      graph_def=gdef,
      input_map={"Reshape":next_element},
      return_elements=out_nodes
    )
    out = out[0].outputs[0]
    outlist.append(out)

  timings=[]

  with tf.Session(graph=g,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    tf.logging.info("Starting Warmup cycle")
    def mergeTraceStr(mdarr):
      tl=timeline.Timeline(mdarr[0][0].step_stats)
      ctf=tl.generate_chrome_trace_format()
      Gtf=json.loads(ctf)
      deltat=mdarr[0][1][1]
      for md in mdarr[1:]:
        tl=timeline.Timeline(md[0].step_stats)
        ctf=tl.generate_chrome_trace_format()
        tmp=json.loads(ctf)
        deltat=0
        Gtf["traceEvents"].extend(tmp["traceEvents"])
        deltat=md[1][1]
        
      #return json.dumps(Gtf,indent=2)
    rmArr=[[tf.RunMetadata(),0] for x in range(20)]
    if timelineName:
      if gfile.Exists(timelineName):
        gfile.Remove(timelineName)
      ttot=int(0)
      tend=time.time()
      for i in range(20):
        tstart=time.time()
        valt = sess.run(outlist,options=run_options,run_metadata=rmArr[i][0])
        tend=time.time()
        rmArr[i][1]=(int(tstart*1.e6),int(tend*1.e6))
      with gfile.FastGFile(timelineName,"a") as tlf:
        tlf.write(mergeTraceStr(rmArr))
    else:
      for i in range(20):
        valt = sess.run(outlist)

    tf.logging.info("Warmup done. Starting real timing")
    num_iters=1
    for i in range(num_loops):
      tstart=time.time()
      for k in range(num_iters):
        print("<=========================")
        val = sess.run(outlist)
        print ((val[0]).shape)
        print ((val[0])[0,:,:,0])
        print("=========================>")
      timings.append((time.time()-tstart)/float(num_iters))
      print("iter ",i," ",timings[-1])
    comp=sess.run(tf.reduce_all(tf.equal(val[0],valt[0])))
    print("Comparison=",comp)
    sess.close()
    tf.logging.info("Timing loop done!")
    #return timings,comp,val[0],None
    return val[0]

def getInputs():
    for x in x_values:
        for y in y_values:
            input_pipe_l.append(test_image[x:(x+tile_size),y:(y+tile_size)])
    input_g = np.asarray(input_pipe_l, dtype=np.float32)
    return input_g

def createOutput(in_patches):
    predicts = in_patches

    final_prob = np.zeros_like(pp_image)        # final array of probabilities
    element_sums = np.zeros_like(pp_image)      # how many times the pixel had something added to it
    for x in range(len(x_values)):
        for y in range(len(y_values)):
            x_idx = x_values[x]
            y_idx = y_values[y]
            temp_pred = predicts[y+(x*len(y_values)), :, :]
            temp_prob = final_prob[ x_idx:(x_idx+tile_size),y_idx:(y_idx+tile_size)]
            final_prob[x_idx:(x_idx+tile_size),y_idx:(y_idx+tile_size)] = temp_pred + temp_prob
            element_sums[x_idx:(x_idx+tile_size),y_idx:(y_idx+tile_size)] = element_sums[x_idx:(x_idx+tile_size),y_idx:(y_idx+tile_size)] + 1
    final_prob = np.divide(final_prob, element_sums)
    seg_pred = np.copy(final_prob)
    values_below = seg_pred < threshold
    values_above = seg_pred >= threshold

    seg_pred[values_below] = 0
    seg_pred[values_above] = 1
    make_image(final_prob, img_name + "_prob.png")  
    #make_image(seg_pred, folder + img_name + str(threshold) + "_pred.png")  

input_all = getInputs()
inp_dims = (32, 28, 28, 1)
input_patches = np.zeros(inp_dims)
out_dims = (4899,28,28)
output_patches = np.zeros(out_dims)
start_offs = 0

trt_graph = trt.create_inference_graph(getGraph(), out_nodes, 
                max_batch_size=32,
                max_workspace_size_bytes=3000000000,
                precision_mode="FP16")

for i in range (0,153):
    end_offs = start_offs + 32
    input_patches[:,:,:,0] = input_all[start_offs:end_offs,:,:]
    results = timeGraph(trt_graph, input_patches)
    output_patches[start_offs:end_offs,:,:] = results[:,:,:,0]
    start_offs = start_offs + 32

createOutput(output_patches)

