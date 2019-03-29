import inferring_the_pde

options = {'mesh_size': [250,250],      # How large is the (regular) 2D-grid of function values for a fixed t.
                                        # Keep mesh_size[0] = mesh_size[1]
           'layers': 9,                 # Layers of the NN. Also counting the initial layer!
           'dt': 0.015,                 # Time discretization. We step dt*(layers - 1) forward in time.
           'batch_size': 1,             # We take a batch of sub-grids in space
           'noise_level': 0.0,          # Can add some noise to the data (not taken 1 to 1, gets multiplied by stddev)
           'downsample_by': 1,          # Size of sub-grids (in space) * downsample_by = mesh_size
           'filter_size' : 5,           # Size of filters to approximate derivatives via FD
           'iterations' : 5,            # How often to repeat optimization in the warmup-step
           'max_order' : 3,             # Max-order of our hidden PDE. Note: max_order < filter_size!
           'boundary_cond' : 'PERIODIC' # Set to 'PERIODIC' if data has periodic bdry condition to use periodic padding
           }

loss = 10e10

a = inferring_the_pde.OptimizerClass(options)

## Repeat the warmup to hopefully find the global optimum while keeping 
## the moment-matrices fixed and thus the number of parameters low
for i in range(options['iterations']):
    coefs_new, _, loss_new = a.optimize_weights(stage = 'WARMUP', iteration = i)
    if loss_new < loss:
        coefs = coefs_new
        loss = loss_new

print('The coefficients are: \n')
for i in range(len(coefs)):
    print('%.8f' % coefs[i])

print('----------------------------------------------')  # End of warmup

## Optimizing for t>0 while partially freeing up the moment-matrices as well
coefs, moment_matrices, _ = a.optimize_weights(stage = 'NORMAL', coefs = coefs, layer = 1)
for l in range(2, options['layers']):
    coefs, moment_matrices, _ = a.optimize_weights(stage = 'NORMAL', coefs = coefs, layer = l,
                                                   moment_matrices = moment_matrices)
