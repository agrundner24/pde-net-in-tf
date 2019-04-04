import inferring_the_pde
import tensorflow as tf
import time

options = {'mesh_size': [250, 250],     # How large is the (regular) 2D-grid of function values for a fixed t.
                                        # Keep mesh_size[0] = mesh_size[1]
           'layers': 6,                 # Layers of the NN. Also counting the initial layer!
           'dt': 0.005,                 # Time discretization. We step dt*(layers - 1) forward in time.
           'batch_size': 24,            # We take a batch of sub-grids in space
           'noise_level': 0.0,          # Can add some noise to the data (not taken 1 to 1, gets multiplied by stddev)
           'downsample_by': 5,          # Size of sub-grids (in space) * downsample_by = mesh_size
           'filter_size': 7,            # Size of filters to approximate derivatives via FD
           'iterations': 30,            # How often to repeat optimization in the warmup-step
           'max_order': 2,              # Max-order of our hidden PDE. Note: max_order < filter_size!
           'boundary_cond': 'PERIODIC'  # Set to 'PERIODIC' if data has periodic bdry condition to use periodic padding
           }

t0 = time.time()

a = inferring_the_pde.OptimizerClass(options)

# Repeat the warmup to hopefully find the global optimum while keeping
# the moment-matrices fixed and thus the number of parameters low
coefs, _, _ = a.optimize_weights(stage='WARMUP', iterations=options['iterations'])

print('----------------------------------------------')  # End of warmup

# Optimizing for t>0 while partially freeing up the moment-matrices as well
with tf.variable_scope('normal_%d' % 1):
    coefs, moment_matrices, _ = a.optimize_weights(stage='NORMAL', coefs=coefs, layer=1)
for l in range(2, options['layers']):
    with tf.variable_scope('normal_%d' % l):
        coefs, moment_matrices, _ = a.optimize_weights(stage='NORMAL', coefs=coefs, layer=l,
                                                       moment_matrices=moment_matrices)

print('Program ran for %d seconds' % (time.time() - t0))