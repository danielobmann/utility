import tensorflow as tf
import keras.backend as K
import numpy as np
import odl.contrib.tensorflow


class ANETT:

    def __init__(self, encoder, decoder, operator, img_height, img_width, sess=None, weights=None, loss='gauss'):
        if sess is None:
            self._sess = K.get_session()
        else:
            self._sess = sess

        # Operators
        self._encoder = encoder
        self._decoder = decoder
        self._input_shape = [tuple([1] + [int(z) for z in s.shape[1:]]) for s in self._decoder.inputs]

        self._shape = operator.range.shape

        self._operator = odl.contrib.tensorflow.as_tensorflow_layer(operator)

        self._img_height = img_height
        self._img_width = img_width

        # Primal variable
        self._x = tf.placeholder(tf.float32, shape=(1, self._img_height, self._img_width, 1))
        self._x_var = tf.Variable(self._x)
        self._x_predicted = self._decoder(self._encoder(self._x))

        # Splitting variable
        self._xi = [tf.placeholder(tf.float32, shape=s) for s in self._input_shape]
        self._xi_var = [tf.Variable(x, dtype=tf.float32) for x in self._xi]
        self._xi_init = self._encoder(self._x)
        self._x_decoded = self._decoder(self._xi_var)

        # Dual variable
        self._dual = [tf.placeholder(tf.float32, shape=s) for s in self._input_shape]
        self._dual_var = [tf.Variable(x, dtype=tf.float32) for x in self._dual]

        # Parametes for minimization
        self._alpha = tf.placeholder(tf.float32)
        self._beta = tf.placeholder(tf.float32)
        self._rho = tf.placeholder(tf.float32)
        self._q = tf.placeholder(tf.float32)
        self._lr = tf.placeholder(tf.float32, name='learning_rate')
        self._data = tf.placeholder(tf.float32)
        self._mom = tf.placeholder(tf.float32)
        self._weights = self._constant_weights() if weights is None else self._decaying_weights()

        # Loss terms for minimization in x direction
        self._regrec = self._regularizer_reconstructable()(x=self._x_var)  # Reconstruction regularizer
        self._reglq = self._regularizer_lq(self._weights)(x=self._x_var)  # Lq regularizer
        self._datafit = self._data_discrepancy(x=self._x_var, data=self._data, loss=loss)  # Data fit error
        self._auglag = self._augmented_lagrangian()(x=self._x_var)  # Augmented Lagrangian

        self._loss_x = self._datafit + self._rho*self._auglag + self._alpha*self._beta*self._regrec
        self._loss_total = self._datafit + self._alpha*self._reglq + self._alpha*self._beta*self._regrec

        # Optimizer for minimization in x
        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._lr)
        self._minimize = self._optimizer.minimize(self._loss_x, var_list=[self._x_var])

        # Shrinkage operator for minimization in xi
        self._enc_x = self._encoder(self._x_var)
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
        temp = np.asarray(x0).reshape((1, self._img_height, self._img_width, 1))
        xi_inp = self._sess.run(self._xi_init, feed_dict={self._x: temp})
        fd = {self._x: temp}
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
        del temp
        pass

    def _update_x_variable(self, feed_dict, niter=100, tol=10**(-5)):
        err = [self._sess.run(self._loss_x, feed_dict=feed_dict)]
        improv = 1
        while (improv > tol) and (len(err) <= niter):
            self._sess.run(self._minimize_momentum, feed_dict=feed_dict)  # make gradient step with momentum
            err.append(self._sess.run(self._loss_x, feed_dict=feed_dict))
            improv = err[-2] - err[-1]  # calculates improvement of loss function

        return improv

    def reconstruct(self, x0, data, niter=10, lr=10**(-3), alpha=10**(-3), beta=10**(-3), rho=10**(-3),
                    niterx=100, mom=0.8, tol=10**(-3)):
        self._variable_initialization(x0=x0)
        fd = {self._data: data,
              self._alpha: alpha,
              self._beta: beta,
              self._rho: rho,
              self._lr: 0,
              self._mom: mom}
        err = [self._sess.run(self._loss_total, feed_dict=fd)]
        tolerances = []
        for it in range(niter):
            fd[self._lr] = lr(it) if callable(lr) else lr

            impro = self._update_x_variable(feed_dict=fd, niter=niterx, tol=tol)  # Step 1: argmin_x
            tolerances.append(impro)
            self._sess.run(self._update_xi_variable, feed_dict=fd)  # Step 2: argmin_xi
            self._sess.run(self._update_dual_variable, feed_dict=fd)  # Step 3: Dual ascent

            err.append(self._sess.run(self._loss_total, feed_dict=fd))  # Calculate loss after iteration

        xout = self._sess.run(self._x_var, feed_dict=fd)
        xout = xout.reshape((self._img_height, self._img_width))
        del fd

        xdec = self._sess.run(self._x_decoded).reshape((self._img_height, self._img_width))
        return xout, xdec, err, tolerances

    def _regularizer_lq(self, w, q=1):
        def reg(x=None, xi=None):
            assert (x is not None) or (xi is not None)
            if xi is None:
                xi = self._encoder(x)
            return K.sum([tf.norm(xi[i]*w[i], ord=q)**q for i in range(len(w))])
        return reg

    def _regularizer_reconstructable(self, p=2):
        def reg(x=None, xi=None):
            if xi is None:
                xi = self._encoder(x)
            xrec = self._decoder(xi)
            return (1./p) * self._norm(x - xrec, p=p)
        return reg

    def add_regularizer(self, reg):
        self._loss_x += reg(self._x_var)
        self._loss_total += reg(self._x_var)
        print("Added addiotional regularizer!", flush=True)
        pass

    def _data_discrepancy(self, x, data, p=2, loss='gauss', mu=0.02, photons=1e4):
        data = tf.reshape(tf.convert_to_tensor(data), (1,) + self._shape + (1,))
        if loss == 'gauss':
            ret = (1./p)*self._norm(self._operator(x) - data, p=p)
        elif loss == 'poisson':
            k_value = (tf.exp(-mu*data)*photons - 1)
            lambda_x = tf.exp(-mu*self._operator(x))*photons
            pois = lambda_x - k_value*tf.log(lambda_x)
            ret = tf.reduce_sum(pois)
        elif loss == 'poisson_approx':
            k_value = (tf.exp(-mu * data) * photons - 1)
            lambda_x = tf.exp(-mu * self._operator(x)) * photons
            ret = tf.log(lambda_x) + (1./lambda_x)*tf.squared_difference(lambda_x, k_value)
            ret = 0.5*tf.reduce_sum(ret)
        elif loss == 'poisson_l2':
            k_value = (tf.exp(-mu * data) * photons - 1)
            lambda_x = tf.exp(-mu * self._operator(x)) * photons
            ret = tf.squared_difference(lambda_x, k_value)/lambda_x
            ret = 0.5*tf.reduce_sum(ret)
        elif loss == 'kl':
            k_value = (tf.exp(-mu * data) * photons - 1)
            k_value = k_value/tf.reduce_sum(k_value)
            lambda_x = tf.exp(-mu * self._operator(x)) * photons
            lambda_x = lambda_x/tf.reduce_sum(lambda_x)

            ret = tf.reduce_sum(lambda_x*tf.log(lambda_x/k_value))
        elif loss == 'mixture':
            # Poisson
            k_value = (tf.exp(-mu * data) * photons - 1)
            lambda_x = tf.exp(-mu * self._operator(x)) * photons
            ret = tf.squared_difference(lambda_x, k_value) / lambda_x
            ret = 0.5 * tf.reduce_sum(ret)

            # l2
            ret += (1. / p) * self._norm(self._operator(x) - data, p=p)

        else:
            ret = tf.zeros(1)
            print("WARNING: No data-discrepancy chosen!", flush=True)
        return ret

    def _augmented_lagrangian(self):
        v = self._list_subtract(self._xi_var, self._dual_var)

        def loss(x):
            xi = self._encoder(x)
            ret = self._list_norm(self._list_subtract(xi, v)) if isinstance(xi, list) else self._norm(xi - v)
            return (1./2)*ret
        return loss

    def _decaying_weights(self):
        w = []
        for s in self._decoder.inputs:
            t = s.shape[1:]
            scale = 2 ** (1 + np.log2(s.shape[1].value) - np.log2(self._img_height))
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
        fd = {self._x: np.asarray(x)[None, ..., None]}
        pred = self._sess.run(self._x_predicted, feed_dict=fd)
        return pred[0, ..., 0]
