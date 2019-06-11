import numpy as np
import numpy.fft as fft

############ Helper functions ############

## Initial value generator

def _initgen_periodic(mesh_size, freq=3):
    #np.random.seed(50)
    # Default: (mesh_size, freq) = ([250, 250], 4)
    dim = len(mesh_size)
    # Default: 250x250-matrix of normally distributed variables
    x = np.random.randn(*mesh_size)
    coe = fft.ifftn(x)
    # set frequency of generated initial value
    # Array of random ints in [freq, 2*freq - 1]
    freqs = np.random.randint(freq, 2*freq, size=[dim,])
    # freqs = [10,10]
    for i in range(dim):
        perm = np.arange(dim, dtype=np.int32)
        perm[i] = 0
        perm[0] = i
        # Permutes for i = 1 and does nothing for i = 0.
        coe = coe.transpose(*perm)
        coe[freqs[i]+1:-freqs[i]] = 0
        coe = coe.transpose(*perm)
    x = fft.fftn(coe)
    assert np.linalg.norm(x.imag) < 1e-8
    x = x.real
    return x
def initgen(mesh_size, freq=3, boundary='Periodic'):
    # Default: (mesh_size, freq, boundary) = ([250, 250], 4, 'Periodic')
    if np.iterable(freq):
        return freq
    # 250x250 normally distributed variables IFFTed and FFTed:
    x = _initgen_periodic(mesh_size, freq=freq)
    x = x*100
    if boundary.upper() == 'DIRICHLET':
        dim = x.ndim
        for i in range(dim):
            y = np.arange(mesh_size[i])/mesh_size[i]
            y = y*(1-y)
            s = np.ones(dim, dtype=np.int32)
            s[i] = mesh_size[i]
            y = np.reshape(y, s)
            x = x*y
        x = x[[slice(1,None),]*dim]
        x = x*16
    return x

## Constructing the mesh

def xy(mesh_size):
    x = 2*np.pi*np.arange(mesh_size[0])/mesh_size[0]
    sample = {}
    sample['x'] = np.repeat(x[np.newaxis,:], mesh_size[0], axis=0)
    sample['y'] = np.repeat(x[:,np.newaxis], mesh_size[0], axis=1)
    return sample

## Data generation for t > 0

class PDESolver(object):
    def step(self, init, dt):
        raise NotImplementedError
    # Note that with the default values (max_dt = 5e-3, T = 0.015), we do fourth-order Runge-Kutta three times
    # each time with a time step 0.005.
    def predict(self, init, T):
        if not hasattr(self, 'max_dt'):
            return self.step(init, T)
        else:
            n = int(np.ceil(T/self.max_dt))
            dt = T/n
            u = init
            for i in range(n):
                u = self.step(u, dt)
            return u

def coe_modify(A, B, m):
    A[:m,:m] = B[:m,:m]
    A[:m,-m+1:] = B[:m,-m+1:]
    A[-m+1:,:m] = B[-m+1:,:m]
    A[-m+1:,-m+1:] = B[-m+1:,-m+1:]
    return

class _variantcoelinear2d(PDESolver):
    # The spectral_size N sets the length of K0, K1 and most importantly the mesh_size, i.e. the 2D-mesh will consist
    # of NxN points.
    def __init__(self, spectral_size, max_dt=5e-3, variant_coe_magnitude=1):
        assert isinstance(spectral_size, int)
        N = spectral_size
        self.max_dt = max_dt
        assert N%2 == 0
        self._N = spectral_size
        self._coe_mag = variant_coe_magnitude
        # Array of N zeros
        freq_shift_coe = np.zeros((N,))
        freq_shift_coe[:N//2] = np.arange(N//2)
        freq_shift_coe[:-N//2-1:-1] = -np.arange(1, 1+N//2)
        # For N = 30: freq_shift_coe = [0, ...,  14, -15, ... ,-1]
        self.K0 = np.reshape(freq_shift_coe, (N,1)) # column 'tensor'
        self.K1 = np.reshape(freq_shift_coe, (1,N)) # row 'tensor'

        # x is of shape 30x30x2 and defines the 2D 30x30-mesh.
        # Filling a 5x5-matrix with functions, that each have a 30x30-matrix as output.
        # Maybe this 5 stems from kernel_size = 5? But I don't know.
        self.a = np.ndarray([5,5], dtype=np.object)
        self.a[0,0] = lambda x:np.zeros(x.shape[:-1])
        self.a[0,1] = lambda x:np.zeros(x.shape[:-1]) + 2  # Represents d/dy
        self.a[1,0] = lambda x:np.zeros(x.shape[:-1]) + 2  # Represents d/dx
        self.a[0,2] = lambda x:np.zeros(x.shape[:-1])      # Probably represents d²/dy²
        self.a[1,1] = lambda x:np.zeros(x.shape[:-1])      # Probably represents d²/d(x,y)
        self.a[2,0] = lambda x:np.zeros(x.shape[:-1])      # Probably represents d²/dx²
        b00 = lambda x:np.zeros(x.shape[:-1])              # Higher derivatives are all zero. In any case, these aren't used for the calculation of a_fourier_coe and a_smooth.
        self.a[list(range(4)),list(range(3,-1,-1))] = b00
        self.a[list(range(5)),list(range(4,-1,-1))] = b00

        # Setting a_fourier_coe and a_smooth
        self.a_fourier_coe = np.ndarray([5,5], dtype=np.object)
        self.a_smooth = np.ndarray([5,5], dtype=np.object)

        xx = np.arange(0,2*np.pi,2*np.pi/N)
        yy = xx.copy()
        yy,xx = np.meshgrid(xx,yy)
        # Inserts a new axis at the given position, shape = 30x30 => shape = 30x30x1
        xx = np.expand_dims(xx, axis=-1)
        yy = np.expand_dims(yy, axis=-1)
        # Join the two arrays along the (second) column axis
        xy = np.concatenate([xx,yy], axis=2)
        m = N//2
        for k in range(3):
            for j in range(k+1):
                # 2-dimensional inverse discrete Fourier Transform
                # tmp_fourier is a NxN-matrix
                tmp_fourier = fft.ifft2(self.a[j,k-j](xy))
                # We fill six values in the upper left triangle of a_fourier_coe with NxN-matrices
                self.a_fourier_coe[j,k-j] = tmp_fourier
                # tmp is some rearrangement of tmp_fourier
                tmp = np.zeros([m*3,m*3], dtype=np.complex128)
                coe_modify(tmp, tmp_fourier, m)
                # Real part of the 2-dimensional discrete Fourier Transform
                # So it'advection_diffusion equivalent to a applied directly on the mesh!
                self.a_smooth[j,k-j] = fft.fft2(tmp).real

    @property
    def spectral_size(self):
        return self._N
    def vc_conv(self, order, coe):
        N = self.spectral_size
        m = N//2
        vc_smooth = self.a_smooth[order[0], order[1]]
        tmp = np.zeros(vc_smooth.shape, dtype=np.complex128)
        coe_modify(tmp, coe, m)
        C_aug = fft.ifft2(vc_smooth*fft.fft2(tmp))
        C = np.zeros(coe.shape, dtype=np.complex128)
        coe_modify(C, C_aug, m)
        return C
    def rhs_fourier(self, L):
        rhsL = np.zeros(L.shape, dtype=np.complex128)
        rhsL += self.vc_conv([1,0], -1j*self.K0*L)
        rhsL += self.vc_conv([0,1], -1j*self.K1*L)
        rhsL += self.vc_conv([2,0], -self.K0**2*L)
        rhsL += self.vc_conv([1,1], -self.K0*self.K1*L)
        rhsL += self.vc_conv([0,2], -self.K1**2*L)
        return rhsL
    # This is doing the step with numerical methods Runge-Kutta 4.
    # We transform the initial data via IFFT, use RK-4, transform it back and take the real part.
    def step(self, init, dt):
        Y = np.zeros([self._N,self._N], dtype=np.complex128)
        m = self._N//2
        L = fft.ifft2(init)
        coe_modify(Y, L, m)
        rhsL1 = self.rhs_fourier(Y)                 # k_1
        rhsL2 = self.rhs_fourier(Y+0.5*dt*rhsL1)    # k_2
        rhsL3 = self.rhs_fourier(Y+0.5*dt*rhsL2)    # k_3
        rhsL4 = self.rhs_fourier(Y+dt*rhsL3)        # k_4

        Y = Y+(rhsL1+2*rhsL2+2*rhsL3+rhsL4)*dt/6
        coe_modify(L, Y, m)
        x_tmp = fft.fft2(L)
        assert np.linalg.norm(x_tmp.imag) < 1e-10
        x = x_tmp.real
        return x



############ Actual Data Generation ############

def downsample(sample, scale):
    # np.random.seed(50)
    idx1 = slice(np.random.randint(scale), None, scale)
    idx2 = slice(np.random.randint(scale), None, scale)
    # idx1 = slice(1, None, scale)
    # idx2 = slice(0, None, scale)

    for kwarg in sample:
        sample[kwarg] = sample[kwarg][idx1, idx2]
    return sample

def addNoise(sample, noise, layers):
    # Adding noise to u0
    mean = sample['u0'].mean()
    stdvar = np.sqrt(((sample['u0'] - mean) ** 2).mean())
    size = sample['u0'].shape
    startnoise = np.random.standard_normal(size)
    sample['u0'] = sample['u0'] + noise * stdvar * startnoise

    # Adding noise to ut, t > 0
    for l in range(1, layers):
        arg = 'u' + str(l)
        size = sample[arg].shape
        endnoise = np.random.standard_normal(size)
        sample[arg] = sample[arg] + noise * stdvar * endnoise

    return sample

def generate(options):

    mesh_size = options['mesh_size']
    layers = options['layers']
    dt = options['dt']
    batch_size = options['batch_size']
    noise_level = options['noise_level']
    downsample_by = options['downsample_by']

    spectral_size = 30

    # Data. Batch contains sample['x'], sample['y'], sample['u0'], sample['u1'], ..., sample['u20']
    batch = []
    for i in range(batch_size):
        sample = {}
        sample['u0'] = initgen(mesh_size, freq=4, boundary='Periodic')
        sample.update(xy(mesh_size))

        pde = _variantcoelinear2d(spectral_size=spectral_size)

        for l in range(1, layers):
            sample['u' + str(l)] = pde.predict(sample['u0'], dt*l)

        sample = downsample(sample, downsample_by)
        sample = addNoise(sample, noise_level, layers)

        batch.append(sample)


        # #Plotting one sample out of curiosity:
        #
        # import matplotlib.pyplot as plt
        # import itertools as it
        #
        # fig, axes = plt.subplots(2, 10)
        # axit = (ax for ax in it.chain(*axes))
        # for l in range(9):
        #     ax = next(axit)
        #     ax.set_title('u' + str(l))
        #     h = ax.imshow(sample['u' + str(l)], interpolation='nearest', cmap='jet',
        #                   extent=[0, 2*np.pi, 0, 2*np.pi],
        #                   origin='lower', aspect='auto')
        #
        # plt.show()

    return batch

