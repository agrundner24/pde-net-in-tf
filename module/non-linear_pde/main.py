import inferring_the_pde
import tensorflow as tf
import time

options = {'mesh_size': [250, 250],     # How large is the (regular) 2D-grid of function values for a fixed t.
                                        # Keep mesh_size[0] = mesh_size[1]
           'layers': 8,                 # Layers of the NN. Also counting the initial layer!
           'dt': 0.003,                 # Time discretization. We step dt*(layers - 1) forward in time.
           'batch_size': 24,            # We take a batch of sub-grids in space
           'noise_level': 0.0,          # Can add some noise to the data (not taken 1 to 1, gets multiplied by stddev)
           'downsample_by': 5,          # Size of sub-grids (in space) * downsample_by = mesh_size
           'filter_size': 7,            # Size of filters to approximate derivatives via FD. Must be an odd number!
           'iterations': 5,             # How often to repeat optimization in the warmup-step
           'max_order': 3,              # Max-order of our hidden PDE. Note: max_order < filter_size!
           'boundary_cond': 'PERIODIC'  # Set to 'PERIODIC' if data has periodic bdry condition to use periodic padding
           }

for q in range(100):

    t0 = time.time()

    a = inferring_the_pde.OptimizerClass(options)

    # Repeat the warmup to hopefully find the global optimum while keeping
    # the moment-matrices fixed and thus the number of parameters low
    coefs, _, param, _ = a.optimize_weights(stage='WARMUP', iterations=options['iterations'])

    print('----------------------------------------------')  # End of warmup

    # Optimizing for t>0 while partially freeing up the moment-matrices as well
    with tf.variable_scope('normal_%d' % 1):
        coefs, moment_matrices, param, _ = a.optimize_weights(stage='NORMAL', coefs=coefs, layer=1, param=param)
    for l in range(2, options['layers']):
        with tf.variable_scope('normal_%d' % l):
            coefs, moment_matrices, param, _ = a.optimize_weights(stage='NORMAL', coefs=coefs, layer=l,
                                                                  moment_matrices=moment_matrices, param=param)

    print('Program ran for %d seconds' % (time.time() - t0))

    with open('Results.txt', 'a') as file:
        file.write('Program ran for %d seconds' % (time.time() - t0))
        file.write('\n' + str(coefs) + '\n')
        file.write('The factor of the non-linear term is %.8f \n' % param)
        file.write('MSE: %.8f \n' % (1/11*(coefs[0] ** 2 + coefs[1] ** 2 + coefs[2] ** 2 + (coefs[3] - 0.3) ** 2
                                      + coefs[4] ** 2 + (coefs[5] - 0.3) ** 2 + coefs[6] ** 2 + coefs[7] ** 2
                                      + coefs[8] ** 2 + coefs[9] ** 2 + (param - 15) ** 2)))

    tf.reset_default_graph()
