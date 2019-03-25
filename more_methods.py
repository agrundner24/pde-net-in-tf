import tensorflow as tf
import numpy as np

# Creating the moment-matrices

def index(filter_size):
    """
    :param filter_size: How large should the filters be
    :param N: N = filter_size * (filter_size + 1) / 2
    :return: 1 + amount of zeros we have before the one in the moment matrices we want to generate
    """
    N = int(filter_size * (filter_size + 1) / 2)
    ind = np.zeros(N)

    for i in range(filter_size):
        start_value = filter_size*i + 1
        start_index = np.argsort(ind)[0] - i

        for j in range(filter_size):
            if start_index + i < N:
                ind[start_index + i] = start_value
            else:
                break
            start_value += 1
            j += 1
            start_index += j + i + 1

    return ind

def multiplier(filter_size, i):
    M = np.ones((filter_size, filter_size))
    i = i + 1

    arr = []
    j = 0
    k = 0
    while k < i:
        k += j
        j += 1
        if k != 0:
            arr.append(k)

    a = np.ones(k)
    a[i - 1] = 0
    spl = np.split(a, arr)

    N = np.zeros((filter_size, filter_size))
    for el in spl:
        N += np.diag(el, filter_size - len(el))
    return np.flip(M - N, 1).T

# Converting a moment-matrix to a filter

def moment_to_filter(M):
    """
    :param M: Moment matrix
    :return:  Filter, that can be used for the convolution
    """
    size = int(M.shape[0])
    # We know: M = P1 * Q * P2 with P1 and P2 defined as follows:
    P1 = np.zeros((size, size))
    P2 = np.zeros((size, size))
    for i in range(size):
        for k1 in range(size):
            P1[i, k1] = 1/np.math.factorial(i)*(k1 - (size - 1)/2)**i
    for k2 in range(size):
        for j in range(size):
            P2[k2, j] = 1/np.math.factorial(j)*(k2 - (size - 1)/2)**j

    return np.linalg.inv(P1) @ M @ np.linalg.inv(P2)

def scaling_factor(i, max_domain, grid_size):
    base = max_domain/grid_size
    exp = 0
    while (exp + 2)*(exp + 1)/2 <= i:
        exp += 1
    return 1/base**exp

# Only to linear_convection momentToFilter:
def filter_to_moment(Q):
    size = int(Q.shape[0])

    M = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            sum = 0
            for k1 in range(size):
                for k2 in range(size):
                    sum += (k1 - (size - 1)/2)**i*(k2 - (size - 1)/2)**j*Q[k1, k2]
            M[i,j] = 1/(np.math.factorial(i)*np.math.factorial(j))*sum

    return M

# for i in range(20):
#     M = 10*np.random.rand(5,5)
#     if np.allclose(momentToFilter(filterToMoment(M)) - M, 0) == False:
#         print('The linear_convection was unsuccessful')
#         break
# print('End of the linear_convection')
#
# M = np.flip(np.diag([1,0,0], 2), 1)
# print(momentToFilter(M))
# print(M)

def pad_input(input, filter_size):
    """
    :param input: Tensor of the shape: batch_size x mesh_size x mesh_size x 1
    :return: We increase the size of input[j,:,:,0] for all j by filter_size // 2 on each side of the matrix
             by inserting values from the opposite side
    """
    batch_size = input.shape[0]
    mesh_size = input.shape[1]
    k = filter_size // 2

    B = np.eye(mesh_size, dtype=np.float32)
    for i in range(k):
        a = np.zeros(mesh_size, dtype=np.float32)
        a[mesh_size - i - 1] = 1
        B = np.concatenate(([a], B), axis=0)
    for i in range(k):
        a = np.zeros(mesh_size, dtype=np.float32)
        a[i] = 1
        B = np.concatenate((B, [a]), axis=0)

    out = []
    C = B.T
    for i in range(batch_size):
        out.append(tf.matmul(tf.matmul(B, input[i, :, :, 0]), C))

    return tf.expand_dims(out, axis = -1)


    # output = np.zeros((batch_size, mesh_size + 2*k, mesh_size + 2*k, 1))
    # output[:, k:mesh_size+k, k:mesh_size+k, :] = input
    # # Setting the first k and last k rows
    # output[:, 0:k, k:mesh_size+k, 0] = input[:, mesh_size-k:mesh_size, :, 0]
    # output[:, mesh_size+k:mesh_size+2*k, k:mesh_size+k, 0] = input[:, 0:k, :, 0]
    # # Setting the first k and last k columns
    # output[:, k:mesh_size+k, 0:k, 0] = input[:, :, mesh_size-k:mesh_size, 0]
    # output[:, k:mesh_size+k, mesh_size+k:mesh_size+2*k, 0] = input[:, :, 0:k, 0]
    # # Now on to the corners:
    # output[:, 0:k, 0:k, 0] = input[:, mesh_size-k:mesh_size, mesh_size-k:mesh_size, 0]
    # output[:, 0:k, mesh_size+k:mesh_size+2*k, 0] = input[:, mesh_size-k:mesh_size, 0:k, 0]
    # output[:, mesh_size+k:mesh_size+2*k, 0:k, 0] = input[:, 0:k, mesh_size-k:mesh_size, 0]
    # output[:, mesh_size+k:mesh_size+2*k, mesh_size+k:mesh_size+2*k, 0] = input[:, 0:k, 0:k, 0]

    # return output

def callback(loss):
    print('Loss: %e' % (loss))
    #print('coef:' + str(coef))
    #print('M:' + str(M))



