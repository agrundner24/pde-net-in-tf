import tensorflow as tf
import numpy as np
import more_methods as mm
import generate_data as gd  # Change this folder to use different data!

np.set_printoptions(linewidth=100)

## For TensorBoard:
# from datetime import datetime
# import sys

# now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
# root_logdir = 'tf_logs'
# logdir = '{}/run-{}/'.format(root_logdir, now)

## For Debugging:
# tf.enable_eager_execution()

class OptimizerClass:

    def __init__(self, options):
        self.options = options
        self.grid_size = int(options['mesh_size'][0] / options['downsample_by'])
        self.max_domain = 2 * np.pi  # Not variable for now. Domain is fixed to [0, 2*pi]x[0, 2*pi]
        self.filter_size = options['filter_size']
        self.batch_size = options['batch_size']
        self.dt = options['dt']
        self.boundary_cond = options['boundary_cond']
        self.max_order = options['max_order']
        self.iterations = options['iterations']

        # Number of filters:
        self.N = int((self.max_order + 2) * (self.max_order + 1) / 2)

        # Positioning the 1 in the moment-matrices
        self.ind = mm.index(self.filter_size)

        self.coefs = []  # Storing the coefficients here
        self.M = []  # Storing the moment-matrices here
        self.param = 0  # Storing additional parameters of the PDE

        # Generating the data
        self.batch = gd.generate(options)

    def set_M(self, M, stage):

        # Either way we cannot work with self.M already being filled with immutable tf.tensors
        self.M = []

        if M is None:
            A = []
            for i in range(self.N):
                B = tf.constant(1, dtype=tf.float32)
                a = int(self.ind[i])
                if stage == 'WARMUP':
                    A.append(tf.constant(np.zeros(self.filter_size ** 2 - 1), dtype=tf.float32, name='A' + str(i)))
                elif stage == 'NORMAL':
                    A.append(tf.Variable(np.zeros(self.filter_size ** 2 - 1), dtype=tf.float32, name='A' + str(i)))
                if a == 0:
                    self.M.append(tf.reshape(tf.concat([[B], A[i]], 0), [self.filter_size, self.filter_size]))
                else:
                    A1, A2 = tf.split(A[i], [a, (self.filter_size ** 2 - 1) - a], 0)
                    M1 = tf.reshape(tf.concat([A1, [B], A2], 0), [self.filter_size, self.filter_size])
                    M2 = mm.multiplier(self.filter_size, i)
                    self.M.append(tf.multiply(M1, M2))  # element-wise multiplication
            # Constructing an additional moment-matrix for the identity-mapping
            if stage == 'WARMUP':
                self.M.append(tf.reshape(tf.concat([[B], tf.constant(np.zeros(self.filter_size ** 2 - 1),
                                                                     dtype=tf.float32)], 0),
                                         [self.filter_size, self.filter_size]))
            elif stage == 'NORMAL':
                self.M.append(tf.reshape(tf.concat([[B], tf.Variable(np.zeros(self.filter_size ** 2 - 1),
                                                                     dtype=tf.float32)], 0),
                                         [self.filter_size, self.filter_size]))

        else:
            # Note that the optimizer.minimize-function does not, when run, change the values of self.M and self.coefs
            # So this assignment of self.M here is crucial (as we do not want to start from the same value in each run)
            assert len(M) in [self.N, self.N + 1]
            for i in range(self.N):
                self.M.append(tf.multiply(tf.get_variable('M%d' % i, initializer=M[i]),
                                          mm.multiplier(self.filter_size, i)))
            # Constructing an additional moment-matrix for the identity-mapping
            Z = np.ones((self.filter_size, self.filter_size))
            Z[0, 0] = 1 / M[self.N][0, 0]
            self.M.append(tf.multiply(tf.get_variable('M%d' % self.N, initializer=M[self.N]), Z))

    def set_coef(self, coefs):
        # Setting up the coefficients that will be learned
        if coefs is None:
            self.coefs = tf.get_variable('coefs', shape=[self.N], initializer=tf.initializers.random_normal)
        else:
            self.coefs = tf.get_variable('coefs', initializer=tf.constant(coefs))

    def optimize_weights(self, stage, coefs=None, layer=None, moment_matrices=None,
                         param=None, iterations=None):
        """
        The key method of this implementation. It contains of a construction (where we set up the neural network)
        and an execution phase, where the actual execution happens.

        :param stage: Either 'WARMUP' (just one layer, with frozen moment-matrices) or 'NORMAL'.
        :param coefs: Previously learned coefficients of the derivative terms in the PDE
        :param layer: Using a convolutional neural network with 'layer' many layers, stepping dt*layer in time.
                      For warmup, no layer has to be provided. Will be set to 1.
        :param moment_matrices: Previously learned moment-matrices corresponding to approx. of derivatives
        :param param: Previously learned additional parameters in the PDE
        :param iterations: How often to repeat optimization in the warmup-step, see options['iterations'] in main.py
        :return: Learned coefficients, parameters and moment-matrices of the PDE with the corresponding loss-value
        """

        # Construction phase
        if stage == 'WARMUP':
            layer = 1

        # Setting up all parameters
        with tf.name_scope('initializing_coefs_and_moment_matrices'):
            self.set_M(moment_matrices, stage)
            self.set_coef(coefs)

            if param is not None:
                self.param = tf.Variable(param, dtype=tf.float32, name='f_param')
            else:
                self.param = tf.Variable(np.random.randint(-200, 200), dtype=tf.float32, name='f_param')

        # Conversion of the moment-matrices into filters.
        with tf.name_scope('moment_to_filter'):
            Q = []
            for i in range(self.N):
                Q.append(mm.moment_to_filter(self.M[i]) * mm.scaling_factor(i, self.max_domain, self.grid_size))
            # Constructing an additional filter for the identity-mapping
            Q.append(mm.moment_to_filter(self.M[self.N]))

        # How the filters will be applied
        with tf.name_scope('convolution'):
            k = self.filter_size // 2

            # Better to stick to pure numpy nd-arrays as long as possible.
            input = np.zeros([self.batch_size, self.grid_size, self.grid_size, 1], dtype=np.float32)

            for j in range(self.batch_size):
                # dtype of the input must match that of the filter and conv2d expects 4D-Tensors
                input[j, :, :, 0] = self.batch[j]['u0'].astype(np.float32)

            # Normalizing factor for the loss-function
            mean = tf.reduce_mean(input)
            var = tf.reduce_mean((input - mean * np.ones(input.shape)) ** 2)

            for l in range(layer):
                out = 0
                f = mm.f(self.param, input)
                if self.boundary_cond == 'PERIODIC':
                    input = mm.pad_input(input, self.filter_size)
                for i in range(self.N):
                    filter = tf.expand_dims(tf.expand_dims(Q[i], axis=-1), -1)
                    out += self.coefs[i] * tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
                input = tf.nn.conv2d(input, tf.expand_dims(tf.expand_dims(Q[self.N], axis=-1), -1),
                                     strides=[1, 1, 1, 1], padding='VALID') + self.dt*(out + f)

        # How the loss will be calculated
        with tf.name_scope('loss'):
            loss = 0
            if self.boundary_cond == 'PERIODIC':
                for j in range(self.batch_size):
                    loss += tf.norm(self.batch[j]['u' + str(layer)] - input[j, :, :, 0], ord='fro',
                                    axis=(0, 1), name='loss') ** 2 / var
            else:
                assert layer * k < self.grid_size - layer * k
                for j in range(self.batch_size):
                    loss += tf.norm(self.batch[j]['u' + str(layer)][layer * k:self.grid_size - layer * k,
                                    layer * k:self.grid_size - layer * k] - input[j, :, :, 0], ord='fro',
                                    axis=(0, 1), name='loss') ** 2 / var

        # Choosing the optimizer
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                           method='L-BFGS-B',
                                                           options={'maxiter': 20000,
                                                                    'maxcor': 500,
                                                                    'gtol': 1e-16,
                                                                    'ftol': 1e0 * np.finfo(float).eps
                                                                    })

        # optimizer_adam = tf.train.AdamOptimizer()
        # training_op_adam = optimizer_adam.minimize(loss)

        ## For TensorBoard:
        # loss_summary = tf.summary.scalar('loss', loss)
        # fileWriter = tf.summary.FileWriter(logdir, tf.get_default_graph())

        init = tf.global_variables_initializer()

        # Otherwise we get issues with CUDNN-initializiation (https://github.com/tensorflow/tensorflow/issues/6698):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Execution phase
        with tf.Session(config=config) as sess:
            loss_comp = 10e10

            ## For TensorBoard:
            # summary_str = loss_summary.eval()
            # fileWriter.add_summary(summary_str)

            # Note that due to the callback the loss gets printed after every LBFGS-iteration
            if stage == "WARMUP":
                for i in range(iterations):
                    sess.run(init)
                    optimizer.minimize(sess, fetches=[loss], loss_callback=mm.callback)

                    if loss.eval() < loss_comp:
                        coefs = self.coefs # Take care cause of div
                        loss_comp = loss.eval()
                        param = self.param

                    print('===========================')

                self.param = param
                self.coefs = coefs

                print('End of warmup')
            else:
                sess.run(init)
                optimizer.minimize(sess, fetches=[loss], loss_callback=mm.callback)
                print('\nEnd of layer %d\n' % layer)

            # Evaluating the loss, moment-matrices, coefficients and possibly PDE-parameters
            coef_out = []
            moment_out = []
            for i in range(self.N):
                # We potentially have to normalize the moment-matrices using div
                div = self.M[i].eval()[int((self.ind[i]) / self.filter_size), int((self.ind[i]) % self.filter_size)]
                coef_out.append(self.coefs[i].eval() * div)
                moment_out.append(self.M[i].eval() / div)
            # The extra moment-matrix corresponding to the identity:
            moment_out.append(self.M[self.N].eval() / self.M[self.N].eval()[0, 0])
            f_param = self.param.eval()
            loss_out = loss.eval()

        # Printing results of this layer
        print('The coefficients are: \n')
        for i in range(self.N):
            print('%.8f' % coef_out[i])

        print('The parameter of f is %.8f' % f_param)

        print('\nThe moment matrices are: \n')
        for i in range(self.N + 1):
            print(moment_out[i])

        return coef_out, moment_out, f_param, loss_out
