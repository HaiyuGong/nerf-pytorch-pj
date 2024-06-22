from load_llff import load_llff_data
import numpy as np


datadir = './data/nerf_llff_data/ania'
factor = 8
spherify = False
llffhold = 8
images, poses, bds, render_poses, i_test = load_llff_data(datadir, factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=spherify)
# print('images: ', images)
i_test = np.arange(images.shape[0])[::llffhold]
print('Test poses', i_test)