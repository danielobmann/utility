import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


def normal_posterior(mu, rho):
    sigma = K.log(1.01 + K.exp(rho))
    return tf.distributions.Normal(mu, sigma)


def normal_prior(mu=0.0, sigma=1e0):
    return tf.distributions.Normal(mu, sigma)

def deconv_length(dim_size, stride_size, kernel_size, padding, output_padding=None, dilation=1):

    assert padding in {'same', 'valid', 'full'}
    if dim_size is None:
        return None

    # Get the dilated kernel size
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)

    # Infer length if output padding is None, else compute the exact length
    if output_padding is None:
        if padding == 'valid':
            dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
        elif padding == 'full':
            dim_size = dim_size * stride_size - (stride_size + kernel_size - 2)
        elif padding == 'same':
            dim_size = dim_size * stride_size
    else:
        if padding == 'same':
            pad = kernel_size // 2
        elif padding == 'valid':
            pad = 0
        elif padding == 'full':
            pad = kernel_size - 1

        dim_size = ((dim_size - 1) * stride_size + kernel_size - 2 * pad + output_padding)

    return dim_size


class BayesianConv2D(Layer):
    def __init__(self, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None,
                 kernel_posterior=None,
                 kernel_prior=None,
                 bias_posterior=None,
                 bias_prior=None,
                 trainable=True,
                 mu_init=keras.initializers.glorot_uniform(),
                 rho_init=keras.initializers.RandomNormal(-5, 1e-2),
                 name=None,
                 use_bias=True,
                 **kwargs):
        super(BayesianConv2D, self).__init__(name=name, **kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.trainable = trainable
        self.use_bias = use_bias

        self.mu_init = mu_init
        self.rho_init = rho_init

        self.kernel_posterior = kernel_posterior
        self.kernel_prior = kernel_prior

        self.bias_posterior = bias_posterior
        self.bias_prior = bias_prior

    def build(self, input_shape):
        # Weights
        self.kernel_shape = self.kernel_size + (int(input_shape[-1]), self.filters)

        # Initialize trainable weights
        self.w_mu = self.add_weight(shape=self.kernel_shape,
                                    initializer=self.mu_init,
                                    trainable=self.trainable,
                                    name="weights_mu")
        self.w_rho = self.add_weight(shape=self.kernel_shape,
                                     initializer=self.rho_init,
                                     trainable=self.trainable,
                                     name="weights_rho")

        if self.kernel_posterior is None:
            self.kernel_posterior = normal_posterior(self.w_mu, self.w_rho)
        if self.kernel_prior is None:
            self.kernel_prior = normal_prior()

        # Bias
        if self.use_bias:
            self.bias_shape = (self.filters,)

            # Initialize trainable biases
            self.b_mu = self.add_weight(shape=self.bias_shape,
                                        initializer=keras.initializers.zeros(),
                                        trainable=self.trainable,
                                        name="biases_mu")

            self.b_rho = self.add_weight(shape=self.bias_shape,
                                         initializer=self.rho_init,
                                         trainable=self.trainable,
                                         name="biases_rho")

            if self.bias_posterior is None:
                self.bias_posterior = normal_posterior(self.b_mu, self.b_rho)
            if self.bias_prior is None:
                self.bias_prior = normal_prior()


    def call(self, inputs):
        w = self.kernel_posterior.sample()
        outputs = K.conv2d(inputs, w, strides=self.strides, padding=self.padding)
        self._apply_divergence(self.kernel_posterior, self.kernel_prior, name='kernel_div')

        if self.use_bias:
            b = self.bias_posterior.sample()
            outputs = K.bias_add(outputs, b)
            self._apply_divergence(self.bias_posterior, self.bias_prior, name='bias_div')

        if self.activation is not None:
            outputs = tf.keras.activations.deserialize(self.activation)(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        # TODO: correct output shape
        if self.padding == 'same':
            output_shape = (
            input_shape[0], input_shape[1] // self.strides[0], input_shape[2] // self.strides[1], self.filters)
        else:
            output_shape = (None,)
        return output_shape

    def _apply_divergence(self, q, p, name=''):
        self.add_loss(tf.reduce_sum(tf.distributions.kl_divergence(q, p), name=name))


class BayesianConv2DTranspose(Layer):
    def __init__(self, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None,
                 kernel_posterior=None,
                 kernel_prior=None,
                 bias_posterior=None,
                 bias_prior=None,
                 trainable=True,
                 mu_init=keras.initializers.glorot_uniform(),
                 rho_init=keras.initializers.RandomNormal(-5, 1e-2),
                 name=None,
                 use_bias=True,
                 **kwargs):
        super(BayesianConv2DTranspose, self).__init__(name=name, **kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.trainable = trainable
        self.use_bias = use_bias

        self.mu_init = mu_init
        self.rho_init = rho_init

        self.kernel_posterior = kernel_posterior
        self.kernel_prior = kernel_prior

        self.bias_posterior = bias_posterior
        self.bias_prior = bias_prior

    def build(self, input_shape):
        # Weights
        self.kernel_shape = self.kernel_size + (self.filters, int(input_shape[-1]))

        # Initialize trainable weights
        self.w_mu = self.add_weight(shape=self.kernel_shape,
                                    initializer=self.mu_init,
                                    trainable=self.trainable,
                                    name="weights_mu")
        self.w_rho = self.add_weight(shape=self.kernel_shape,
                                     initializer=self.rho_init,
                                     trainable=self.trainable,
                                     name="weights_rho")

        if self.kernel_posterior is None:
            self.kernel_posterior = normal_posterior(self.w_mu, self.w_rho)
        if self.kernel_prior is None:
            self.kernel_prior = normal_prior()

        # Bias
        if self.use_bias:
            self.bias_shape = (self.filters,)

            # Initialize trainable biases
            self.b_mu = self.add_weight(shape=self.bias_shape,
                                        initializer=keras.initializers.zeros(),
                                        trainable=self.trainable,
                                        name="biases_mu")

            self.b_rho = self.add_weight(shape=self.bias_shape,
                                         initializer=self.rho_init,
                                         trainable=self.trainable,
                                         name="biases_rho")

            if self.bias_posterior is None:
                self.bias_posterior = normal_posterior(self.b_mu, self.b_rho)
            if self.bias_prior is None:
                self.bias_prior = normal_prior()


    def call(self, inputs):

        w = self.kernel_posterior.sample()
        inp_shape = K.int_shape(inputs)
        bs = tf.shape(inputs)[0]
        out_shape = self.compute_output_shape(inp_shape)
        out_shape = tf.stack([bs, out_shape[1], out_shape[2], out_shape[3]])
        outputs = K.conv2d_transpose(inputs, w,
                                     output_shape=out_shape,
                                     strides=self.strides,
                                     padding=self.padding)
        self._apply_divergence(self.kernel_posterior, self.kernel_prior, name='kernel_div')

        if self.use_bias:
            b = self.bias_posterior.sample()
            outputs = K.bias_add(outputs, b)
            self._apply_divergence(self.bias_posterior, self.bias_prior, name='bias_div')

        if self.activation is not None:
            outputs = tf.keras.activations.deserialize(self.activation)(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        out_height = deconv_length(int(input_shape[1]), self.strides[0], self.kernel_size[0], self.padding)
        out_width = deconv_length(int(input_shape[2]), self.strides[1], self.kernel_size[1], self.padding)

        output_shape = (input_shape[0], out_height, out_width, self.filters)
        return output_shape

    def _apply_divergence(self, q, p, name=''):
        self.add_loss(tf.reduce_sum(tf.distributions.kl_divergence(q, p), name=name))

# TODO: Implement BayesianBatchNormalization, BayesianDense, ...


class BayesianDense(Layer):
    def __init__(self, size=128, activation=None,
                 weight_posterior=None,
                 weight_prior=None,
                 bias_posterior=None,
                 bias_prior=None,
                 trainable=True,
                 mu_init=keras.initializers.glorot_uniform(),
                 rho_init=keras.initializers.RandomNormal(-5, 1e-2),
                 name=None,
                 use_bias=True,
                 **kwargs):
        super(BayesianDense, self).__init__(name=name, **kwargs)
        self.size = size
        self.activation = activation
        self.use_bias = use_bias
        self.trainable = trainable
        self.use_bias = use_bias

        self.mu_init = mu_init
        self.rho_init = rho_init

        self.weight_posterior = weight_posterior
        self.weight_prior = weight_prior

        self.bias_posterior = bias_posterior
        self.bias_prior = bias_prior

    def build(self, input_shape):
        # Weights
        self.weight_shape = (int(input_shape[-1]), self.size)

        # Initialize trainable weights
        self.w_mu = self.add_weight(shape=self.weight_shape,
                                    initializer=self.mu_init,
                                    trainable=self.trainable,
                                    name="weights_mu")
        self.w_rho = self.add_weight(shape=self.weight_shape,
                                     initializer=self.rho_init,
                                     trainable=self.trainable,
                                     name="weights_rho")

        if self.weight_posterior is None:
            self.weight_posterior = normal_posterior(self.w_mu, self.w_rho)
        if self.weight_prior is None:
            self.weight_prior = normal_prior()

        # Bias
        if self.use_bias:
            self.bias_shape = (self.size,)

            # Initialize trainable biases
            self.b_mu = self.add_weight(shape=self.bias_shape,
                                        initializer=keras.initializers.zeros(),
                                        trainable=self.trainable,
                                        name="biases_mu")

            self.b_rho = self.add_weight(shape=self.bias_shape,
                                         initializer=self.rho_init,
                                         trainable=self.trainable,
                                         name="biases_rho")

            if self.bias_posterior is None:
                self.bias_posterior = normal_posterior(self.b_mu, self.b_rho)
            if self.bias_prior is None:
                self.bias_prior = normal_prior()


    def call(self, inputs):
        w = self.weight_posterior.sample()

        outputs = K.dot(inputs, w)
        self._apply_divergence(self.weight_posterior, self.weight_prior, name='kernel_div')

        if self.use_bias:
            b = self.bias_posterior.sample()
            outputs = K.bias_add(outputs, b)
            self._apply_divergence(self.bias_posterior, self.bias_prior, name='bias_div')

        if self.activation is not None:
            outputs = tf.keras.activations.deserialize(self.activation)(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.size)
        return output_shape

    def _apply_divergence(self, q, p, name=''):
        self.add_loss(tf.reduce_sum(tf.distributions.kl_divergence(q, p), name=name))


class BayesianPReLU(Layer):

    def __init__(self, alpha_initializer='zeros', alpha_posterior=None, alpha_prior=None, **kwargs):
        super(BayesianPReLU, self).__init__(**kwargs)
        self.alpha_initializer = keras.initializers.get(alpha_initializer)
        self.alpha_posterior = alpha_posterior
        self.alpha_prior = alpha_prior

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.alpha_mu = self.add_weight(shape=param_shape, name='alpha', initializer=self.alpha_initializer)
        self.alpha_rho = self.add_weight(shape=param_shape, name='alpha', initializer=self.alpha_initializer)

        if self.alpha_posterior is None:
            self.alpha_posterior = normal_posterior(self.alpha_mu, self.alpha_rho)

        if self.alpha_prior is None:
            self.alpha_prior = normal_prior()

        self.built = True

    def call(self, inputs):
        alpha = self.alpha_posterior.sample()
        pos = K.relu(inputs)
        neg = -alpha * K.relu(-inputs)
        self._apply_divergence(self.alpha_posterior, self.alpha_prior, name='alpha_div')
        return pos + neg

    def compute_output_shape(self, input_shape):
        return input_shape

    def _apply_divergence(self, q, p, name=''):
        self.add_loss(tf.reduce_sum(tf.distributions.kl_divergence(q, p), name=name))


class BayesianModel:
    def __init__(self, architecture):
        self.architecture = architecture
        self.loss = 0

    def __call__(self, inputs):
        out = inputs
        for i in range(len(self.architecture)):
            out = self.architecture[i](out)
            try:
                self.loss += sum(self.architecture[i].losses)
            except AttributeError:
                pass
        return out