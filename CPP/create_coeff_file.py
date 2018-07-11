import sys

import numpy as np
from scipy import misc

img_no = '01'
img_name = img_no + "_test"
img_path =   'Images/' + img_name + ".tif"

print("Saving coefficients for " + img_path)

test_image = np.asarray(misc.imread(img_path), dtype=np.float32) 
coeff_path =   'data/all_coeff_' + img_name + ".txt"
file = open(coeff_path, 'w')        
for i in range(0,584):
    for j in range(0, 565):
        file.write(str(test_image[i,j]) + '\n')
file.close()

print("Created coefficients file " + coeff_path);
