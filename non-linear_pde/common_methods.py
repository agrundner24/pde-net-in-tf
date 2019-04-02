import numpy as np
import numpy.fft as fft


def pad_input_2(input, pad_by):
    """
    We increase the size of input for all j by pad_by on each side of the matrix
    by inserting values from the opposite side
    """
    mesh_size = input.shape[0]

    B = np.eye(mesh_size, dtype=np.float32)
    for i in range(pad_by):
        a = np.zeros(mesh_size, dtype=np.float32)
        a[mesh_size - i - 1] = 1
        B = np.concatenate(([a], B), axis=0)
    for i in range(pad_by):
        a = np.zeros(mesh_size, dtype=np.float32)
        a[i] = 1
        B = np.concatenate((B, [a]), axis=0)

    return B @ input @ B.T


## The following methods are pretty much the same as in the original PDE-Net ##

def downsample(sample, scale):
    """
    Returns a regular somewhat random sub-grid of sample, whose size is reduced by a factor of 'scale'.
    """

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


############ Initial value generator ############

def initgen(mesh_size, freq=3, boundary='Periodic'):
    """
    Returns function values for t=0 on a regular grid of size 'mesh_size' in [0, 2*pi]x[0, 2*pi] as a matrix
    """
    # Default: (mesh_size, freq, boundary) = ([250, 250], 4, 'Periodic')
    if np.iterable(freq):
        return freq
    # 250x250 normally distributed variables IFFTed and FFTed:
    x = _initgen_periodic(mesh_size, freq=freq)
    x = x * 100
    if boundary.upper() == 'DIRICHLET':
        dim = x.ndim
        for i in range(dim):
            y = np.arange(mesh_size[i]) / mesh_size[i]
            y = y * (1 - y)
            s = np.ones(dim, dtype=np.int32)
            s[i] = mesh_size[i]
            y = np.reshape(y, s)
            x = x * y
        x = x[[slice(1, None), ] * dim]
        x = x * 16
    return x


###################################################


def _initgen_periodic(mesh_size, freq=3):
    # np.random.seed(50)
    # Default: (mesh_size, freq) = ([250, 250], 4)
    dim = len(mesh_size)
    # Default: 250x250-matrix of normally distributed variables
    x = np.random.randn(*mesh_size)
    coe = fft.ifftn(x)
    # set frequency of generated initial value
    # Array of random ints in [freq, 2*freq - 1]
    freqs = np.random.randint(freq, 2 * freq, size=[dim, ])
    # freqs = [10,10]
    for i in range(dim):
        perm = np.arange(dim, dtype=np.int32)
        perm[i] = 0
        perm[0] = i
        # Permutes for i = 1 and does nothing for i = 0.
        coe = coe.transpose(*perm)
        coe[freqs[i] + 1:-freqs[i]] = 0
        coe = coe.transpose(*perm)
    x = fft.fftn(coe)
    assert np.linalg.norm(x.imag) < 1e-8
    x = x.real
    return x
