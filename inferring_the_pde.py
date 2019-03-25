import tensorflow as tf
import numpy as np
import time
import more_methods as mM
import cfd_python.advection_diffusion.generate_data as gD   # Change the folder for different data
from datetime import datetime
# import sys
# tf.enable_eager_execution()
np.set_printoptions(linewidth=100)
############################

# now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
# root_logdir = 'tf_logs'
# logdir = '{}/run-{}/'.format(root_logdir, now)

############################

class OptimizerClass():

    def __init__(self, options):
        self.options = options
        self.grid_size = int(options['mesh_size'][0]/options['downsample_by'])
        self.max_domain = 2*np.pi        # Not variable for now. Domain is fixed to [0, 2*pi]x[0, 2*pi]
        self.filter_size = options['filter_size']
        self.batch_size = options['batch_size']
        self.dt = options['dt']
        self.boundary_cond = options['boundary_cond']

        # Number of filters:
        self.N = int(self.filter_size * (self.filter_size + 1) / 2)

        # Positioning the 1 in the moment-matrices
        self.ind = mM.index(self.filter_size)

        self.coefs = []          # Storing the coefficients here
        self.M = []              # Storing the moment-matrices here

        # Generating the data
        self.batch = gD.generate(options)


    def set_M(self, M, stage):

        # Either way we cannot work with self.M already being filled with immutable tf.tensors
        self.M = []

        if M == None:
            A = []
            for i in range(self.N):
                B = tf.constant(1, dtype=tf.float32)
                a = int(self.ind[i] - 1)
                if stage == 'WARMUP':
                    A.append(tf.constant(np.zeros(self.filter_size ** 2 - 1), dtype=tf.float32, name='A' + str(i)))
                elif stage == 'NORMAL':
                    A.append(tf.Variable(np.zeros(self.filter_size ** 2 - 1), dtype=tf.float32, name='A' + str(i)))
                # Now take only a part of A
                if a == 0:
                    self.M.append(tf.reshape(tf.concat([[B], A[i]], 0), [self.filter_size, self.filter_size]))
                else:
                    A1, A2 = tf.split(A[i], [a, (self.filter_size ** 2 - 1) - a], 0)
                    M1 = tf.reshape(tf.concat([A1, [B], A2], 0), [self.filter_size, self.filter_size])
                    M2 = mM.multiplier(self.filter_size, i)
                    self.M.append(tf.multiply(M1, M2))  # element-wise multiplication
            # Constructing an additional moment-matrix for the identity-mapping
            if stage == 'WARMUP':
                self.M.append(tf.reshape(tf.concat([[B], tf.constant(np.zeros(self.filter_size ** 2 - 1),
                                        dtype=tf.float32)], 0), [self.filter_size, self.filter_size]))
            elif stage == 'NORMAL':
                self.M.append(tf.reshape(tf.concat([[B], tf.Variable(np.zeros(self.filter_size ** 2 - 1),
                                        dtype=tf.float32)], 0), [self.filter_size, self.filter_size]))

        else:
            # Note that the optimizer.minimize-function does not, when run, change the values of self.M and self.coefs
            # So this assignment of self.M here is crucial (as we do not want to start from the same value in each run)
            assert len(M) in [self.N, self.N + 1]
            for i in range(self.N):
                 self.M.append(tf.multiply(tf.Variable(M[i]), mM.multiplier(self.filter_size, i)))
            # Constructing an additional moment-matrix for the identity-mapping
            Z = np.ones((self.filter_size, self.filter_size))
            Z[0,0] = 1/M[self.N][0,0]
            self.M.append(tf.multiply(tf.Variable(M[self.N]), Z))

    def set_coef(self, coefs):
        if coefs == None and len(self.coefs) == 0:
            for i in range(self.N):
                self.coefs.append(tf.Variable(np.random.randn(1), dtype=tf.float32, name='coef' + str(i)))
        else:
            self.coefs = []
            for i in range(self.N):
                self.coefs.append(tf.Variable(coefs[i], dtype=tf.float32, name='coef' + str(i)))


    def optimize_weights(self, stage, coefs = None, layer = None, moment_matrices = None):
        if stage == 'WARMUP':
            layer = 1

        with tf.name_scope('initializing_coefs_and_moment_matrices') as scope:
            self.set_M(moment_matrices, stage)
            self.set_coef(coefs)

        # Converting the moment-matrices into filters. conv2D wants the transpose of the filter.
        with tf.name_scope('moment_to_filter') as scope:
            Q = []
            for i in range(self.N):
                Q.append(mM.moment_to_filter(self.M[i])*mM.scaling_factor(i, self.max_domain, self.grid_size))
            # Constructing an additional filter for the identity-mapping
            Q.append(mM.moment_to_filter(self.M[self.N]))

        # Applying the filters and calculating the loss
        # Start with only one layer
        with tf.name_scope('convolution') as scope:
            k = self.filter_size // 2

            # Better to stick with pure numpy nd-arrays as long as possible.
            input = np.zeros([self.batch_size, self.grid_size, self.grid_size, 1], dtype = np.float32)

            for j in range(self.batch_size):
                # dtype of the input must match that of the filter and conv2d expects 4D-Tensors
                # Assign doesn't assign the value but creates a tf.Operation that can be run in a session!...
                input[j,:,:,0] = self.batch[j]['u0'].astype(np.float32)

            for l in range(1, layer + 1):
                out = 0
                if self.boundary_cond == 'PERIODIC':
                    input = mM.pad_input(input, self.filter_size)
                for i in range(self.N):
                    filter = tf.expand_dims(tf.expand_dims(Q[i], axis=-1), -1)
                    out += self.coefs[i] * tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
                input = out * self.dt + tf.nn.conv2d(input, tf.expand_dims(tf.expand_dims(Q[self.N], axis=-1), -1), \
                             strides=[1, 1, 1, 1], padding='VALID')


        # Computing the loss
        with tf.name_scope('loss') as scope:
            loss = 0
            if self.boundary_cond == 'PERIODIC':
                for j in range(self.batch_size):
                    loss += tf.norm(self.batch[j]['u' + str(layer)] - input[j,:,:,0], ord = 'fro', \
                                    axis=(0,1), name='loss')
            else:
                assert layer*k < self.grid_size-layer*k
                for j in range(self.batch_size):
                    loss += tf.norm(self.batch[j]['u' + str(layer)][layer*k:self.grid_size-layer*k, \
                                    layer*k:self.grid_size-layer*k] - input[j,:,:,0], ord = 'fro', \
                                    axis=(0,1), name='loss')


        # Which optimizer we choose
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss,
                       method = 'L-BFGS-B',
                       options = {'maxiter': 20000,
                                  'maxfun': 50000,
                                  'maxcor': 50,
                                  'maxls': 50,
                                  'ftol': 1*np.finfo(float).eps
                                  })

        # optimizer_adam = tf.train.AdamOptimizer()
        # training_op_adam = optimizer_adam.minimize(loss)

        init = tf.global_variables_initializer()

        # loss_summary = tf.summary.scalar('loss', loss)
        # fileWriter = tf.summary.FileWriter(logdir, tf.get_default_graph())


        # Execution phase
        t0 = time.time()

        with tf.Session() as sess:
            sess.run(init)

            # summary_str = loss_summary.eval()
            # fileWriter.add_summary(summary_str)

            optimizer.minimize(sess, fetches=[loss], loss_callback=mM.callback)

            if stage == 'WARMUP':
                print('End of warmup')
            else:
                print('End of layer %d' % layer)

            coef_out = []
            moment_out = []
            for i in range(self.N):
                # We potentially have to normalize the moment-matrices using div
                div = self.M[i].eval()[int((self.ind[i]-1)/self.filter_size), int((self.ind[i]-1) % self.filter_size)]
                coef_out.append(self.coefs[i].eval()*div)
                moment_out.append(self.M[i].eval()/div)
            moment_out.append(self.M[self.N].eval())

        print('Time used for training: ' + str(time.time() - t0) + '\n')

        print('The coefficients are: \n')
        for i in range(self.N):
            print('%.8f' % coef_out[i])

        print('\nThe moment matrices are: \n')
        for i in range(self.N + 1):
            print(moment_out[i])

        return coef_out, moment_out