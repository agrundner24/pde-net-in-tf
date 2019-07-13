import numpy as np
import common_methods as com
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from mpl_toolkits.mplot3d import Axes3D

def generate(options, coefs=[0, -1, -1, 0.3, 0, 0.3, 0, 0, 0, 0], downsample = True, method = 'orig', init = None):
    """
    Generating data / function-values on a regular grid of space-time, adding noise and taking a batch of
    down-sampled regular sub-grids of this grid. This batch will contain the samples to train our network with.

    :param options: The dictionary of user-specified options (cf. main.py). Contains e.g. the grid-dimensions and noise
    :return: A batch (as a list) of samples (as dictionaries), that in turn consist of (noisy) function values on
             down-sampled sub-grids for all dt-layers.
    """

    # Variable declarations
    nx = options['mesh_size'][0]
    ny = options['mesh_size'][1]
    nt = options['layers']
    dt = options['dt']
    noise_level = options['noise_level']
    if downsample is True:
        downsample_by = options['downsample_by']
        batch_size = options['batch_size']
    else:
        downsample_by = 1
        batch_size = 1

    dx = 2*np.pi/(nx - 1)
    dy = 2*np.pi/(ny - 1)

    # # Needed for plotting:
    # x = np.linspace(0, 2*np.pi, num = nx)
    # y = np.linspace(0, 2*np.pi, num = ny)
    # X, Y = np.meshgrid(x, y)

    batch = []
    inits = []

    for i in range(batch_size):
        ########################### Change the following lines to implement your own data ###########################

        sample = {}
        ## Assign initial function:
        if init is not None:
            u = init
        elif method == 'orig':
            u = com.initgen(options['mesh_size'], freq=4, boundary='Periodic')
        elif method == 'rbf':
            u = com.initgen_custom_rbf(options['mesh_size'])
        elif method == 'wavelet':
            u = com.initgen_custom_wavelet(options['mesh_size'])
        elif method == 'poly':
            u = com.initgen_custom_order2pol(options['mesh_size'])
        elif method == 'low_freq':
            u = com.initgen(options['mesh_size'], freq=2, boundary='Periodic')


        sample['u0'] = u[(nt - 1)*2:(nx - (nt - 1)*2), (nt - 1)*2:(ny - (nt - 1)*2)]
        inits.append(u)

        # # Plotting the initial function:
        # fig = plt.figure(figsize=(11,7), dpi=100)
        # ax = fig.gca(projection='3d')
        # surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
        #
        # plt.show()

        for n in range(nt - 1):
            un = u

            u = (un[2:-2, 2:-2] + dt * (coefs[0] * un[2:-2, 2:-2]
                                        + coefs[1] * un[2:-2, 2:-2] * (un[2:-2, 3:-1] - un[2:-2, 2:-2]) / dy
                                        + coefs[2] * un[2:-2, 2:-2] * (un[3:-1, 2:-2] - un[2:-2, 2:-2]) / dx
                                        + coefs[3] * (un[2:-2, 3:-1] + un[2:-2, 1:-3] - 2 * un[2:-2, 2:-2]) / dy ** 2
                                        + coefs[4] * (un[3:-1, 3:-1] - un[3:-1, 1:-3] - un[1:-3, 3:-1]+un[1:-3, 1:-3])
                                                    / (4 * dx * dy)
                                        + coefs[5] * (un[3:-1, 2:-2] + un[1:-3, 2:-2] - 2 * un[2:-2, 2:-2]) / dx ** 2
                                        + coefs[6] * (2 * un[2:-2, 1:-3] - 2 * un[2:-2, 3:-1] - un[2:-2, :-4]
                                                      + un[2:-2, 4:]) / (2 * dy ** 3)
                                        + coefs[7] * (-un[3:-1, 4:] + 16 * un[3:-1, 3:-1] - 30 * un[3:-1, 2:-2]
                                                      + 16 * un[3:-1, 1:-3] - un[3:-1, :-4] + un[1:-3, 4:]
                                                      - 16 * un[1:-3, 3:-1] + 30 * un[1:-3, 2:-2] - 16 * un[1:-3, 1:-3]
                                                      + un[1:-3, :-4]) / (24 * dx * dy ** 2)
                                        + coefs[8] * (-un[4:, 3:-1] + 16 * un[3:-1, 3:-1] - 30 * un[2:-2, 3:-1]
                                                      + 16 * un[1:-3, 3:-1] - un[:-4, 3:-1] + un[4:, 1:-3]
                                                      - 16 * un[3:-1, 1:-3] + 30 * un[2:-2, 1:-3] - 16 * un[1:-3, 1:-3]
                                                      + un[:-4, 1:-3]) / (24 * dy * dx ** 2)
                                        + coefs[9] * (2 * un[1:-3, 2:-2] - 2 * un[3:-1, 2:-2] - un[:-4, 2:-2]
                                                      + un[4:, 2:-2]) / (2 * dx ** 3)))

            # The data of all layers should have the same size!
            sample['u' + str(n + 1)] = u[2*((nt - 1) - n - 1):(u.shape[0] - 2*((nt - 1) - n - 1)),
                                       2*((nt - 1) - n - 1):(u.shape[1] - 2*((nt - 1) - n - 1))]


        ## sample should at this point be a dictionary with entries 'u0', ..., 'uL', where L = nt                   ##
        ## For a given j, sample['uj'] is a matrix of size nx x ny containing the function values at time-step dt*j ##
        ##############################################################################################################


        # # Plotting the function values from the last layer:
        # fig2 = plt.figure()
        # ax2 = fig2.gca(projection='3d')
        # surf2 = ax2.plot_surface(X[10:260, 10:260], Y[10:260, 10:260], u, cmap=cm.viridis)
        #
        # plt.show()

        com.downsample(sample, downsample_by)
        com.addNoise(sample, noise_level, nt)

        batch.append(sample)

    if batch_size == 1:
        return batch[0], inits[0]
    else:
        return batch, inits

