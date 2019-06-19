import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import common_methods as com

options = {'mesh_size': [250, 250],     # How large is the (regular) 2D-grid of function values for a fixed t.
                                        # Keep mesh_size[0] = mesh_size[1]
           'layers': 6,                 # Layers of the NN. Also counting the initial layer!
           'dt': 0.005,                 # Time discretization. We step dt*(layers - 1) forward in time.
           'batch_size': 1,            # We take a batch of sub-grids in space
           'noise_level': 0.0,          # Can add some noise to the data (not taken 1 to 1, gets multiplied by stddev)
           'downsample_by': 1,          # Size of sub-grids (in space) * downsample_by = mesh_size
           'filter_size': 7,            # Size of filters to approximate derivatives via FD
           'iterations': 5,             # How often to repeat optimization in the warmup-step
           'max_order': 3,              # Max-order of our hidden PDE. Note: max_order < filter_size!
           'boundary_cond': 'PERIODIC'  # Set to 'PERIODIC' if data has periodic bdry condition to use periodic padding
}

# Coefficients from the best run of Burgers' equation in ConvNet
coefs = [-0.0011892784, -1.0029587, -0.9947339, 0.29654312, -0.0006035583, 0.3017471, 0.00059628644, 0.0027602008, 0.0006023276, -0.0020996542]

# Initial function
u_init = com.initgen(options['mesh_size'], freq=4, boundary='Periodic')


def generate_old(options, u):
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

    return batch


def generate(options, coefs, u):
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
    downsample_by = options['downsample_by']
    batch_size = options['batch_size']

    dx = 2*np.pi/(nx - 1)
    dy = 2*np.pi/(ny - 1)

    # Needed for plotting:
    x = np.linspace(0, 2*np.pi, num = nx)
    y = np.linspace(0, 2*np.pi, num = ny)
    X, Y = np.meshgrid(x, y)

    batch = []

    for i in range(batch_size):

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
                                        + coefs[6] * (2*un[2:-2, 1:-3] - 2*un[2:-2, 3:-1] - un[2:-2, :-4] + un[2:-2, 4:])/(2*dx**3)
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

        batch.append(sample)

    return batch


# Applying FD to the learned PDE
batch = generate(options, coefs,  u_init)
# Applying FD to the original PDE
batch_old = generate_old(options, u_init)


# Evaluating the error
print(np.max(batch[0]['u5'] - batch_old[0]['u5']))  # This is the sup-norm