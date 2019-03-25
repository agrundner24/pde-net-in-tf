import numpy as np
import numpy.fft as fft
from .. import common_methods as com
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D

def generate(options):

    # Variable declarations
    nx = options['mesh_size'][0]
    ny = options['mesh_size'][1]
    nt = options['layers']
    dt = options['dt']
    noise_level = options['noise_level']
    downsample_by = options['downsample_by']
    batch_size = options['batch_size']

    nu = 5   # Can see that multiple layers are beneficial when setting nu = 5

    dx = 2*np.pi/(nx - 1)
    dy = 2*np.pi/(ny - 1)

    # x = np.linspace(0, 2*np.pi, num = nx)
    # y = np.linspace(0, 2*np.pi, num = ny)
    # X, Y = np.meshgrid(x, y)
    #
    # u = np.ones((ny, nx))
    # un = np.ones((ny, nx))

    # Assign initial conditions
    # u[int(0.5/dy):int(1/dy) + 1, int(0.5/dy):int(1/dy) + 1] = 2

    u = com.initgen(options['mesh_size'], freq=4, boundary='Periodic')

    # fig = plt.figure(figsize=(11,7), dpi=100)
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
    #
    # plt.show()

    sample = {}
    # sample['u0'] = u
    sample['u0'] = u

    for n in range(nt - 1):
        un = u.copy()

        u[1:-1, 1:-1] = un[1:-1,1:-1] + nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + \
                        nu * dt / dy**2 * (un[2:,1: -1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])

        u = com.pad_input_2(u[1:-1, 1:-1], 1)

        sample['u' + str(n+1)] = u

        # fig2 = plt.figure()
        # ax2 = fig2.gca(projection='3d')
        # surf2 = ax2.plot_surface(X, Y, u, cmap=cm.viridis)
        #
        # plt.show()


    batch = []

    for i in range(batch_size):

        sample_tmp = sample.copy()
        com.downsample(sample_tmp, downsample_by)
        com.addNoise(sample_tmp, noise_level, nt)

        batch.append(sample_tmp)

    return batch
