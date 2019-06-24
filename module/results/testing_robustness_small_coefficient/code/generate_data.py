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

    # u_t + u*u_x + u*u_y = nu*(u_{xx} + u_{yy})

    # Variable declarations
    nx = options['mesh_size'][0]
    ny = options['mesh_size'][1]
    nt = options['layers']
    dt = options['dt']
    noise_level = options['noise_level']
    downsample_by = options['downsample_by']
    batch_size = options['batch_size']

    nu = 1e-4

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

        ## Assign initial function:
        u = com.initgen(options['mesh_size'], freq=4, boundary='Periodic')

        inits.append(u)

        ## Plotting the initial function:
        # fig = plt.figure(figsize=(11,7), dpi=100)
        # ax = fig.gca(projection='3d')
        # surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
        #
        # plt.show()

        sample = {}
        sample['u0'] = u

        for n in range(nt - 1):
            un = com.pad_input_2(u, 2)[1:, 1:]  # Same triplet of numbers on each side

            u = (un[1:-1, 1:-1] + dt * (nu*(un[2:, 1:-1] + un[0:-2, 1:-1] - 2*un[1:-1, 1:-1]) / dx**2
                                        + nu*(un[1:-1, 2:] + un[1:-1, 0:-2] - 2*un[1:-1, 1:-1]) / dy**2
                                        - un[1:-1, 1:-1] * (un[2:, 1:-1] - un[1:-1, 1:-1]) / dx
                                        - un[1:-1, 1:-1] * (un[1:-1, 2:] - un[1:-1, 1:-1]) / dy))[:-1, :-1]

            sample['u' + str(n+1)] = u


        ## sample should at this point be a dictionary with entries 'u0', ..., 'uL', where L = nt                   ##
        ## For a given j, sample['uj'] is a matrix of size nx x ny containing the function values at time-step dt*j ##
        ##############################################################################################################


        # # Plotting the function values from the last layer:
        # fig2 = plt.figure()
        # ax2 = fig2.gca(projection='3d')
        # surf2 = ax2.plot_surface(X, Y, u, cmap=cm.viridis)
        #
        # plt.show()

        com.downsample(sample, downsample_by)
        com.addNoise(sample, noise_level, nt)

        batch.append(sample)

    return batch, inits


# To test robustness:
def generate_new(options, coefs, u, downsample_by):
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

    dx = 2*np.pi/(nx - 1)
    dy = 2*np.pi/(ny - 1)

    # # Needed for plotting:
    # x = np.linspace(0, 2*np.pi, num = nx)
    # y = np.linspace(0, 2*np.pi, num = ny)
    # X, Y = np.meshgrid(x, y)

    batch = []

    ## Plotting the initial function:
    # fig = plt.figure(figsize=(11,7), dpi=100)
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
    #
    # plt.show()

    sample = {}
    sample['u0'] = u

    for n in range(nt - 1):
        un = com.pad_input_2(u, 2)

        u = (un[2:-2, 2:-2] + dt * (coefs[5]*(un[3:-1, 2:-2] + un[1:-3, 2:-2] - 2*un[2:-2, 2:-2]) / dx**2
                                    + coefs[3]*(un[2:-2, 3:-1] + un[2:-2, 1:-3] - 2*un[2:-2, 2:-2]) / dy**2
                                    + coefs[2] * un[2:-2, 2:-2] * (un[3:-1, 2:-2] - un[2:-2, 2:-2]) / dx
                                    + coefs[1] * un[2:-2, 2:-2] * (un[2:-2, 3:-1] - un[2:-2, 2:-2]) / dy
                                    + coefs[0] * un[2:-2, 2:-2]
                                    + coefs[4] * (un[3:-1, 3:-1] - un[3:-1, 1:-3] - un[1:-3, 3:-1] + un[1:-3, 1:-3])/(4*dx*dy)
                                    + coefs[6] * (2*un[2:-2, 1:-3] - 2*un[2:-2, 3:-1] - un[2:-2, :-4] + un[2:-2, 4:])/(2*dy**3)
                + coefs[7] * (-un[3:-1, 4:] + 16*un[3:-1, 3:-1] - 30*un[3:-1, 2:-2] + 16*un[3:-1, 1:-3] - un[3:-1, :-4] + un[1:-3, 4:] - 16*un[1:-3, 3:-1] + 30*un[1:-3, 2:-2] - 16*un[1:-3, 1:-3] + un[1:-3, :-4])/(24*dx*dy**2)
                + coefs[8] * (-un[4:, 3:-1] + 16*un[3:-1, 3:-1] - 30*un[2:-2, 3:-1] + 16*un[1:-3, 3:-1] - un[:-4, 3:-1] + un[4:, 1:-3] - 16*un[3:-1, 1:-3] + 30*un[2:-2, 1:-3] - 16*un[1:-3, 1:-3] + un[:-4, 1:-3])/(24*dy*dx**2)
                                    + coefs[9] * (2*un[1:-3, 2:-2] - 2*un[3:-1, 2:-2] - un[:-4, 2:-2] + un[4:, 2:-2])/(2*dx**3)))

        sample['u' + str(n+1)] = u


    # # Plotting the function values from the last layer:
    # fig2 = plt.figure()
    # ax2 = fig2.gca(projection='3d')
    # surf2 = ax2.plot_surface(X, Y, u, cmap=cm.viridis)

    # plt.figure()
    # plt.pcolor(X, Y, u, cmap='jet')
    # plt.colorbar()
    # plt.xlabel('x')
    # plt.ylabel('y')

    # plt.show()

    com.downsample(sample, downsample_by)
    com.addNoise(sample, noise_level, nt)

    return sample