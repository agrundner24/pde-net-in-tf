import numpy as np
import inferring_the_pde

options = {'mesh_size': [250,250],      # Keep mesh_size[0] = mesh_size[1]
           'layers': 9,                 # Also counting the initial layer
           'dt': 0.015,
           'batch_size': 10,
           'noise_level': 0.0,
           'downsample_by': 5,
           'filter_size' : 5,           # Have to make some changes to mM.index to get filter_size != 5 to work
           'boundary_cond' : 'PERIODIC' # Put in anything but periodic to get no padding
           }


a = inferring_the_pde.OptimizerClass(options)

coefs = a.optimize_weights(stage = 'WARMUP')[0]
coefs, moment_matrices = a.optimize_weights(stage = 'NORMAL', coefs = coefs, layer = 1)
for l in range(2, options['layers']):
    coefs, moment_matrices = a.optimize_weights(stage = 'NORMAL', coefs = coefs, layer = l, \
                                            moment_matrices = moment_matrices)