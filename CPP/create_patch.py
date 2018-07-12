import sys

import numpy as np
import matplotlib.pyplot as plt

img_no = '01'
img_name = img_no + "_test"

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

img_in = sys.argv[1];
with open(img_in) as f:
    floats = map(float, f)
img_ch1 = np.reshape(floats[0:784], (28,28))
#img_ch2 = np.reshape(floats[784:], (28,28))
print(type(img_ch1[0,0]))
print(img_ch1.shape)
make_image(img_ch1, img_name + "_" + img_in[:-4] + "_out_ch1.png")
#make_image(img_ch2, img_name + "_" + img_in[:-4] + "_out_ch2.png")
