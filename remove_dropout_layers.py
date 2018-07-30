# This script accepts a frozen tensorflow graph and attempts to remove dropout layers so the model can be used for inference
# Usage: python remove_dropout_layers.py frozen_graph_input_path.pb frozen_graph_output_path.pb
# Based on answers from https://stackoverflow.com/questions/40358892/wipe-out-dropout-operations-from-tensorflow-graph 

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.tools.optimize_for_inference_lib import node_from_map, node_name_from_input

def print_graph(input_graph):
    for node in input_graph.node:
        print "{0} : {1} ( {2} )".format(node.name, node.op, node.input)

def strip(input_graph):
    input_nodes = input_graph.node
    input_node_map = {}
    for node in input_nodes:
      if node.name not in input_node_map.keys():
        input_node_map[node.name] = node
    nodes_after_strip = []

    for node in input_nodes:
      print "======================"
      print "{0} : {1} ( {2} )".format(node.name, node.op, node.input)
      # If node belongs to the dropout layer - ignore it
      if len(node.name.split('/')) > 2 and node.name.split('/')[-3].startswith('dropout'):
          continue

      # Otherwise, copy the node to new graph
      new_node = node_def_pb2.NodeDef()
      new_node.CopyFrom(node)
      for idx, i in enumerate(node.input):
        input_clean = node_name_from_input(i)
        # If node accepts output from dropout layer as its input - change node's input to be dropout layer's input
        if len(input_clean.split('/')) > 2 and input_clean.split('/')[-3].startswith('dropout'):
          op_mul = node_from_map(input_node_map, i)
          op_div = node_from_map(input_node_map, op_mul.input[0])
          parent = node_from_map(input_node_map, op_div.input[0])
          new_node.input[idx] = parent.name
      nodes_after_strip.append(new_node)

    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes_after_strip)
    return output_graph

input_graph = sys.argv[1]
output_graph = sys.argv[2]

input_graph_def = tf.GraphDef()
with tf.gfile.FastGFile(input_graph, "rb") as f:
        input_graph_def.ParseFromString(f.read())

output_graph_def = strip(input_graph_def)

#print "======================\nBefore:"
#print_graph(input_graph_def)

#print "======================\nAfter:"
#print_graph(output_graph_def)

with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

print("%d ops in the original graph." % len(input_graph_def.node))
print("%d ops in the final graph." % len(output_graph_def.node))
