import numpy as np
import common_methods as com
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from mpl_toolkits.mplot3d import Axes3D

def generate(options):
    """
    Generating data / function-values on a regular grid of space-time, adding noise and taking a batch of
    down-sampled regular sub-grids of this grid. This batch will contain the samples to train our network with.

    :param options: The dictionary of user-specified options (cf. main.py). Contains e.g. the grid-dimensions and noise
    :return: A batch (as a list) of samples (as dictionaries), that in turn consist of (noisy) function values on
             down-sampled sub-grids for all dt-layers.
    """

    # Diffusion with non-linear source-term 15*sin(u) (cf. PDE-Net)

    # Variable declarations
    nx = options['mesh_size'][0]
    ny = options['mesh_size'][1]
    nt = options['layers']
    dt = options['dt']
    noise_level = options['noise_level']
    downsample_by = options['downsample_by']
    batch_size = options['batch_size']

    nu = 0.3

    dx = 2*np.pi/(nx - 1)
    dy = 2*np.pi/(ny - 1)

    # # Needed for plotting:
    # x = np.linspace(0, 2*np.pi, num = nx)
    # y = np.linspace(0, 2*np.pi, num = ny)
    # X, Y = np.meshgrid(x, y)

    batch = []

    for i in range(batch_size):
        ########################### Change the following lines to implement your own data ###########################

        ## Assign initial function:
        u = com.initgen(options['mesh_size'], freq=4, boundary='Periodic')

        # ## Plotting the initial function:
        # fig = plt.figure(figsize=(11,7), dpi=100)
        # ax = fig.gca(projection='3d')
        # surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)

        # plt.show()

        sample = {}
        sample['u0'] = u

        for n in range(nt - 1):
            un = com.pad_input_2(u, 2)[1:, 1:]  # Same triplet of numbers on each side

            u = (un[1:-1,1:-1] + nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
                 + nu * dt / dy**2 * (un[2:,1: -1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
                 + dt * 15*np.sin(un[1:-1,1:-1]))[:-1, :-1]

            sample['u' + str(n+1)] = u


        ## sample should at this point be a dictionary with entries 'u0', ..., 'uL', where L = nt                   ##
        ## For a given j, sample['uj'] is a matrix of size nx x ny containing the function values at time-step dt*j ##
        ##############################################################################################################

        com.downsample(sample, downsample_by)
        com.addNoise(sample, noise_level, nt)

        batch.append(sample)

    return batch
