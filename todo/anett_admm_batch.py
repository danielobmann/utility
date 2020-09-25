import tensorflow as tf
import keras.backend as K
import numpy as np
import odl.contrib.tensorflow


class ANETT:

    def __init__(self, encoder, decoder, operator, batch_size=8, size=512, sess=None, weights=None, loss='gauss'):
        if sess is None:
            self._sess = K.get_session()
        else:
            self._sess = sess

        # Operators
        self._encoder = encoder
        self._decoder = decoder
        # Could be made better with K.int_shape
        self._input_shape = [tuple([batch_size] + [int(z) for z in s.shape[1:]]) for s in self._decoder.inputs]

        self._shape = operator.range.shape
        self._operator = odl.contrib.tensorflow.as_tensorflow_layer(operator)
        self._size = size

        # Primal variable
        self._x = tf.placeholder(tf.float32, shape=(batch_size, self._size, self._size, 1))
        self._x_var = tf.Variable(self._x)
        self._x_predicted = self._decoder(self._encoder(self._x))

        self._enc_x = self._encoder(self._x_var)
        self._x_decoded = self._decoder(self._enc_x)

        # Splitting variable
        self._xi = self._encoder(self._x)
        self._xi_var = [tf.Variable(x, dtype=tf.float32) for x in self._xi]

        # Dual variable
        self._dual = [tf.placeholder(tf.float32, shape=s) for s in self._input_shape]
        self._dual_var = [tf.Variable(x, dtype=tf.float32) for x in self._dual]

        # Parametes for minimization
        self._alpha = tf.placeholder(tf.float32)
        self._beta = tf.placeholder(tf.float32)
        self._rho = tf.placeholder(tf.float32)
        self._q = tf.placeholder(tf.float32)
        self._lr = tf.placeholder(tf.float32, name='learning_rate')
        self._data = tf.placeholder(tf.float32, (batch_size, ) + operator.range.shape + (1,))
        self._mom = tf.placeholder(tf.float32)
        self._weights = self._constant_weights() if weights is None else self._decaying_weights()

        # Loss terms for minimization in x direction
        self._regrec = self._regularizer_reconstructable()  # Reconstruction regularizer
        self._reglq = self._regularizer_lq()  # Lq regularizer
        self._datafit = self._data_discrepancy(loss=loss)  # Data fit error
        self._auglag = self._augmented_lagrangian()  # Augmented Lagrangian

        self._loss_x = self._datafit + self._rho*self._auglag + self._alpha*self._beta*self._regrec
        self._loss_total = self._datafit + self._alpha*self._reglq + self._alpha*self._beta*self._regrec

        # Optimizer for minimization in x
        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._lr)
        self._minimize = self._optimizer.minimize(self._loss_x, var_list=[self._x_var])

        # Shrinkage operator for minimization in xi

        self._update_xi_variable = self._xi_update()
        self._constraint_error = self._list_norm(self._list_subtract(self._enc_x, self._xi_var))

        # Dual ascent update
        self._update_dual_variable = self._dual_update()

        # Optimizer with momentum for minimization in x
        self._optimizer_momentum = tf.train.MomentumOptimizer(learning_rate=self._lr,
                                                              momentum=self._mom,
                                                              use_nesterov=True)
        self._minimize_momentum = self._optimizer_momentum.minimize(self._loss_x, var_list=[self._x_var])

        # Variable initializer for all variables!
        self._var_init = tf.variables_initializer([self._x_var] + self._xi_var + self._dual_var +
                                                  self._optimizer_momentum.variables())
        print("ANETT initialization successful!", flush=True)
        pass

    def _xi_update(self):
        if isinstance(self._enc_x, list):
            ret = [z.assign(self._shrinkage(e + u, (self._alpha/self._rho)*w)) for e, z, u, w in
                                    zip(self._enc_x, self._xi_var, self._dual_var, self._weights)]
        else:
            ret = self._xi_var[0].assign(self._shrinkage(self._enc_x + self._dual_var[0], (self._alpha/self._rho)*self._weights[0]))
        return ret

    def _dual_update(self):
        if isinstance(self._enc_x, list):
            ret = [u.assign(u+e-xi) for u, e, xi in zip(self._dual_var, self._enc_x, self._xi_var)]
        else:
            ret = self._dual_var[0].assign(self._dual_var[0] + self._enc_x - self._xi_var[0])
        return ret

    def _variable_initialization(self, x0):
        fd = {self._x: x0}
        xi_inp = self._sess.run(self._xi, feed_dict=fd)

        if isinstance(xi_inp, list):
            for i in range(len(xi_inp)):
                fd[self._xi[i].name] = xi_inp[i]  # initialize xi[i] as E(x)[i]
                fd[self._dual[i].name] = np.zeros(self._input_shape[i])  # initialize u[i] as zero
        else:
            fd[self._xi[0].name] = xi_inp
            fd[self._dual[0].name] = np.zeros(self._input_shape[0])

        self._sess.run(self._var_init, feed_dict=fd)
        del xi_inp
        del fd
        pass

    def _update_x_variable(self, feed_dict, niter=100, tol=10**(-5)):
        err = [self._sess.run(self._loss_x, feed_dict=feed_dict)]
        # Improv has to be remade to avoid weird stuff happening because of batches
        for i in range(niter):
            self._sess.run(self._minimize_momentum, feed_dict=feed_dict)  # make gradient step with momentum
            err.append(self._sess.run(self._loss_x, feed_dict=feed_dict))
        pass

    def reconstruct(self, x0, data, niter=10, lr=10**(-3), alpha=10**(-3), beta=10**(-3), rho=10**(-3),
                    niterx=100, mom=0.8, tol=10**(-3)):
        self._variable_initialization(x0=x0)
        fd = {self._data: data[..., None],
              self._alpha: alpha,
              self._beta: beta,
              self._rho: rho,
              self._lr: 0,
              self._mom: mom}
        err = [self._sess.run(self._loss_total, feed_dict=fd)]

        for it in range(niter):
            fd[self._lr] = lr(it) if callable(lr) else lr

            self._update_x_variable(feed_dict=fd, niter=niterx, tol=tol)  # Step 1: argmin_x
            self._sess.run(self._update_xi_variable, feed_dict=fd)  # Step 2: argmin_xi
            self._sess.run(self._update_dual_variable, feed_dict=fd)  # Step 3: Dual ascent

            err.append(self._sess.run(self._loss_total, feed_dict=fd))  # Calculate loss after iteration

        xout = self._sess.run(self._x_var, feed_dict=fd)
        xdec = self._sess.run(self._x_decoded)
        return xout, xdec, err

    def _regularizer_lq(self, q=1):
        return K.sum([tf.norm(self._enc_x[i], ord=q)**q for i in range(len(self._enc_x))])

    def _regularizer_reconstructable(self, p=2):
        return (1./p) * self._norm(self._x_var - self._x_decoded, p=p)

    def add_regularizer(self, reg):
        self._loss_x += reg(self._x_var)
        self._loss_total += reg(self._x_var)
        print("Added addiotional regularizer!", flush=True)
        pass

    def _data_discrepancy(self, p=2, loss='gauss', mu=0.02, photons=1e4):
        if loss == 'gauss':
            ret = (1./p)*self._norm(self._operator(self._x_var) - self._data, p=p)
        elif loss == 'poisson':
            k_value = (tf.exp(-mu*self._data)*photons - 1)
            lambda_x = tf.exp(-mu*self._operator(self._x_var))*photons
            pois = lambda_x - k_value*tf.log(lambda_x)
            ret = tf.reduce_sum(pois)
        elif loss == 'poisson_approx':
            k_value = (tf.exp(-mu * self._data) * photons - 1)
            lambda_x = tf.exp(-mu * self._operator(self._x_var)) * photons
            ret = tf.log(lambda_x) + (1./lambda_x)*tf.squared_difference(lambda_x, k_value)
            ret = 0.5*tf.reduce_sum(ret)
        elif loss == 'poisson_l2':
            k_value = (tf.exp(-mu * self._data) * photons - 1)
            lambda_x = tf.exp(-mu * self._operator(self._x_var)) * photons
            ret = tf.squared_difference(lambda_x, k_value)/lambda_x
            ret = 0.5*tf.reduce_sum(ret)
        elif loss == 'kl':
            k_value = (tf.exp(-mu * self._data) * photons - 1)
            lambda_x = tf.exp(-mu * self._operator(self._x_var)) * photons

            ret = tf.reduce_sum(lambda_x*tf.log(lambda_x/k_value) - lambda_x)
        elif loss == 'mixture':
            # Poisson
            k_value = (tf.exp(-mu * self._data) * photons - 1)
            lambda_x = tf.exp(-mu * self._operator(self._x_var)) * photons
            ret = tf.squared_difference(lambda_x, k_value) / lambda_x
            ret = 0.5 * tf.reduce_sum(ret)

            # l2
            ret += (1. / p) * self._norm(self._operator(self._x_var) - self._data, p=p)

        else:
            ret = tf.zeros(1)
            print("WARNING: No data-discrepancy chosen!", flush=True)
        return ret

    def _augmented_lagrangian(self):
        v = self._list_subtract(self._xi_var, self._dual_var)
        ret = self._list_norm(self._list_subtract(self._enc_x, v))
        return 0.5*ret

    def _decaying_weights(self):
        w = []
        for s in self._decoder.inputs:
            t = s.shape[1:]
            scale = 2 ** (1 + np.log2(s.shape[1].value) - np.log2(self._size))
            w.append(np.ones([1, ] + [z.value for z in t]) * scale)
        return w

    def _constant_weights(self):
        return [np.ones([1, ] + [z.value for z in s.shape[1:]]) for s in self._decoder.inputs]

    @staticmethod
    def _shrinkage(xi, gamma):
        return tf.maximum(tf.abs(xi) - gamma, 0) * tf.sign(xi)

    @staticmethod
    def _norm(x, p=2):
        """
        Implementation of p-norm to the power of p. This is used in optimization since tf.norm is numerically
        instable for x = 0.
        """
        return K.sum(K.pow(K.abs(x), p))

    @staticmethod
    def _list_subtract(a, b):
        if isinstance(a, list):
            ret = [i - j for i, j in zip(a, b)]
        else:
            ret = a - b
        return ret

    def _list_norm(self, a, p=2):
        if isinstance(a, list):
            ret = K.sum([self._norm(i, p=p) for i in a])
        else:
            ret = self._norm(a, p=p)
        return ret

    def predict(self, x):
        fd = {self._x: np.asarray(x)[..., None]}
        pred = self._sess.run(self._x_predicted, feed_dict=fd)
        return pred
