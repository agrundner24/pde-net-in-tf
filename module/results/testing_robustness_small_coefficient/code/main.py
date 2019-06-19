import inferring_the_pde
import tensorflow as tf
import time
import generate_data as gd
import numpy as np
import matplotlib.pyplot as plt

options = {'mesh_size': [250, 250],     # How large is the (regular) 2D-grid of function values for a fixed t.
                                        # Keep mesh_size[0] = mesh_size[1]
           'layers': 6,                 # Layers of the NN. Also counting the initial layer!
           'dt': 0.005,                 # Time discretization. We step dt*(layers - 1) forward in time.
           'batch_size': 24,            # We take a batch of sub-grids in space
           'noise_level': 0.0,          # Can add some noise to the data (not taken 1 to 1, gets multiplied by stddev)
           'downsample_by': 5,          # Size of sub-grids (in space) * downsample_by = mesh_size
           'filter_size': 7,            # Size of filters to approximate derivatives via FD
           'iterations': 5,             # How often to repeat optimization in the warmup-step
           'max_order': 3,              # Max-order of our hidden PDE. Note: max_order < filter_size!
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


# To test robustness:
def test_robustness(options, coefs, a):
    batch_new = []
    batch_old = []
    for i in range(options['batch_size']):
        batch_old.append(gd.generate_new(options, [0, -1, -1, 0.3, 0, 0.3, 0, 0, 0, 0], a.inits[i], 1))
        batch_new.append(gd.generate_new(options, coefs, a.inits[i], 1))
    return batch_old, batch_new

def test_robustness_with_downsampling(options, coefs, a):
    batch_new = []
    for i in range(options['batch_size']):
        batch_new.append(gd.generate_new(options, coefs, a.inits[i], 5))
    # Here, batch_old is a.batch
    return batch_new

batch_old, batch_new = test_robustness(options, coefs, a)
batch_new_downsampled = test_robustness_with_downsampling(options, coefs, a)

# Test results - no downsampling
# Sup
with open('results_sup', 'a') as file:
    file.write('Results from one batch with 24 samples.\nInferred coefficients: ' + str(coefs)
               + '\n  t = 0  | t=0.005 | t=0.01  | t=0.015 | t=0.02  | t=0.025\n')
    file.write('='*63 + '\n')
    for i in range(options['batch_size']):
            file.write('%8.3f |%8.3f |%8.3f |%8.3f |%8.3f |%8.3f\n' %
                       tuple([np.max(np.abs((batch_old[i]['u%d'% j] - batch_new[i]['u%d'% j]))) for j in range(6)]))
    file.write('\n\n')
# Squared
with open('results_squared', 'a') as file:
    file.write('Results from one batch with 24 samples.\nInferred coefficients: ' + str(coefs)
               + '\n  t = 0  | t=0.005 | t=0.01  | t=0.015 | t=0.02  | t=0.025\n')
    file.write('='*63 + '\n')
    for i in range(options['batch_size']):
            file.write('%8.3f |%8.3f |%8.3f |%8.3f |%8.3f |%8.3f\n' %
                       tuple([np.sum((batch_old[i]['u%d'% j] - batch_new[i]['u%d'% j])**2)/(250**2) for j in range(6)]))
    file.write('\n\n')
# Sum
with open('results_sum', 'a') as file:
    file.write('Results from one batch with 24 samples.\nInferred coefficients: ' + str(coefs)
               + '\n  t = 0  | t=0.005 | t=0.01  | t=0.015 | t=0.02  | t=0.025\n')
    file.write('='*63 + '\n')
    for i in range(options['batch_size']):
            file.write('%8.3f |%8.3f |%8.3f |%8.3f |%8.3f |%8.3f\n' %
                       tuple([np.sum(np.abs((batch_old[i]['u%d'% j] - batch_new[i]['u%d'% j])))/(250**2) for j in range(6)]))
    file.write('\n\n')


# Test results - with downsampling
# Sup
with open('results_downsampled_sup', 'a') as file:
    file.write('Results from one batch with 24 samples.\nInferred coefficients: ' + str(coefs)
               + '\n  t = 0  | t=0.005 | t=0.01  | t=0.015 | t=0.02  | t=0.025\n')
    file.write('='*63 + '\n')
    for i in range(options['batch_size']):
            file.write('%8.3f |%8.3f |%8.3f |%8.3f |%8.3f |%8.3f\n' %
                       tuple([np.max(np.abs((a.batch[i]['u%d'% j] - batch_new_downsampled[i]['u%d'% j]))) for j in range(6)]))
    file.write('\n\n')
# Squared
with open('results_downsampled_squared', 'a') as file:
    file.write('Results from one batch with 24 samples.\nInferred coefficients: ' + str(coefs)
               + '\n  t = 0  | t=0.005 | t=0.01  | t=0.015 | t=0.02  | t=0.025\n')
    file.write('='*63 + '\n')
    for i in range(options['batch_size']):
            file.write('%8.3f |%8.3f |%8.3f |%8.3f |%8.3f |%8.3f\n' %
                       tuple([np.sum((a.batch[i]['u%d'% j] - batch_new_downsampled[i]['u%d'% j])**2)/(50**2) for j in range(6)]))
    file.write('\n\n')
# Sum
with open('results_downsampled_sum', 'a') as file:
    file.write('Results from one batch with 24 samples.\nInferred coefficients: ' + str(coefs)
               + '\n  t = 0  | t=0.005 | t=0.01  | t=0.015 | t=0.02  | t=0.025\n')
    file.write('='*63 + '\n')
    for i in range(options['batch_size']):
            file.write('%8.3f |%8.3f |%8.3f |%8.3f |%8.3f |%8.3f\n' %
                       tuple([np.sum(np.abs((a.batch[i]['u%d'% j] - batch_new_downsampled[i]['u%d'% j])))/(50**2) for j in range(6)]))
    file.write('\n\n')


def plot_solution(u, sample_num, isNew, isGood, t):
    # lb = X_star.min(0)
    # ub = X_star.max(0)
    # nn = 200
    nx = options['mesh_size'][0]
    ny = options['mesh_size'][1]
    x = np.linspace(0, 2*np.pi, num = nx)
    y = np.linspace(0, 2*np.pi, num = ny)
    X, Y = np.meshgrid(x, y)

    # U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')

    plt.figure()
    plt.pcolor(X, Y, u, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')

    if isGood is True:
        if isNew is True:
            plt.savefig('./figures/Burgers_%d_learned_t=%.3f.png' % (sample_num, t))
        else:
            plt.savefig('./figures/Burgers_%d_t=%.3f.png' % (sample_num, t))
    else:
        if isNew is True:
            plt.savefig('./figures/Burgers_%d_learned_fail_t=%.3f.png' % (sample_num, t))
        else:
            plt.savefig('./figures/Burgers_%d_fail_t=%.3f.png' % (sample_num, t))

# Plotting results that are not downsampled
for i in range(24):
    # Plotting a good result (Sum < 0.02)
    if np.sum(np.abs((batch_old[i]['u5'] - batch_new[i]['u5']))) / (250 ** 2) < 0.3:
        for j in range(6):
            plot_solution(batch_old[i]['u%d' % j], sample_num=i, isNew=False, isGood=True, t=0.005*j)
        for j in range(6):
            plot_solution(batch_new[i]['u%d' % j], sample_num=i, isNew=True, isGood=True, t=0.005*j)
    # Plotting a bad result (Sum > 25)
    if np.sum(np.abs((batch_old[i]['u5'] - batch_new[i]['u5']))) / (250 ** 2) > 10:
        for j in range(6):
            plot_solution(batch_old[i]['u%d' % j], sample_num=i, isNew=False, isGood=False, t=0.005*j)
        for j in range(6):
            plot_solution(batch_new[i]['u%d' % j], sample_num=i, isNew=True, isGood=False, t=0.005*j)

