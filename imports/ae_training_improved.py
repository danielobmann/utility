import os
from keras.models import Model
from keras.layers import Input, Lambda, Multiply
import tensorflow as tf
import numpy as np
from PIL import Image

from keras.callbacks import Callback
import warnings


class AutoencoderTraining:
    def __init__(self, decoder, encoder):
        self.decoder = decoder
        self.encoder = encoder

        self.decoder.name = 'decoder_output'
        self.encoder.name = 'encoder_output'

    '''For the implementation of the loss functions, keep in mind that y_pred will be a list given by 
    the elements [prediction of autoencoder, encoded signal, delta]. Here the encoded signal may itself be
    multiple elements.

    For the loss function, we want to calculate the mean only along the first axis (batch size), because
    otherwise this may lead to some unexpected scaling. Consider the case of input images of size (512, 512) and
    the encoded signal is of size (8,8,64). Then the mean of the l2-loss is scaled by 1/(512^2) and the
    l1-loss is scaled like 1/(64^2).
    
    This class allows one to use a custom phi functional for the training of the autoencoder. Additionally,
    it lets you specify what delta should be.'''

    @staticmethod
    def _tf_norm(x, p=2):
        x = tf.cast(x, dtype=tf.float32)
        x = tf.abs(x) ** p
        # By choosing to only take the mean over the first axis we only take the mean over the batch.
        # This is done to avoid any unwanted scaling of the loss function if the encoded output has a different
        # dimension than the input/output of the autoencoder.
        return (1. / p) * tf.reduce_sum(tf.reduce_mean(x, axis=0))

    def _l2_loss(self, y_true, y_pred):
        return self._tf_norm(y_true - y_pred, p=2)

    @staticmethod
    def _phi_loss(alpha):
        def loss(y_true, y_pred):
            return alpha*tf.reduce_sum(y_pred)
        return loss

    @staticmethod
    def _l1_phi(xi):
        if isinstance(xi, list):
            ret = tf.add_n([tf.reduce_sum(tf.abs(z), axis=[1, 2, 3]) for z in xi])
        else:
            ret = tf.reduce_sum(tf.abs(xi), axis=[1, 2, 3])
        return ret

    def network_compile(self, alpha, metrics=[], phi=None, **kwargs):
        # Set up network for training and pass input through network for calculating the loss
        inp = self.encoder.inputs
        out_encoder = self.encoder(inp)
        out_decoder = self.decoder(out_encoder)

        # Delta controls whether the phi-regularization is applied or not.
        # In the case that the input is noise free, we choose delta = 1 and delta = 0 otherwise.
        # This ensures that only the noise-free images are mapped to something sparse!
        inp_delta = Input(shape=(1,))

        phi = self._l1_phi if phi is None else phi
        phi_encoder = Lambda(lambda xi: phi(xi), name='phi')(out_encoder)
        phi_encoder = Multiply(name='phi_output')([inp_delta, phi_encoder])

        output = [out_decoder, phi_encoder]
        network_train = Model(inputs=inp + [inp_delta], outputs=output)

        # Compile network
        losses = {'decoder_output': self._l2_loss,
                  'phi_output': self._phi_loss(alpha)}

        network_train.compile(**kwargs, loss=losses, metrics={'decoder_output': metrics})

        # Return network
        return network_train

    @staticmethod
    def get_data_generator(batch_size, path, mode='train', input_processor=lambda x: x, delta_range=2):
        p = path + '/' + mode
        while True:
            files = np.random.choice(os.listdir(p), size=batch_size, replace=True)
            imgs = [np.asarray(Image.open(p + '/' + file).convert('L')) / 255. for file in files]

            img_inputs = []
            delta_inputs = []

            for i in imgs:
                delta_inputs += [np.random.choice(delta_range)]
                img_inputs += [i if delta_inputs[-1] == 1 else input_processor(i)]

            img_labels = np.stack(imgs)[..., None]
            img_inputs = np.stack(img_inputs)[..., None]
            delta_inputs = np.stack(delta_inputs)

            input = [img_inputs, delta_inputs]
            output = [img_labels, np.zeros(batch_size)]

            yield (input, output)


class AutoencoderCP(Callback):

    def __init__(self, encoder, decoder, encoder_path, decoder_path, monitor='val_loss', verbose=0,
                 mode='auto', period=1):
        super(AutoencoderCP, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.period = period
        self.epochs_since_last_save = 0
        self.encoder = encoder
        self.decoder = decoder

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, ' 'fallback to auto mode.' % (mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, ' 'skipping.' % self.monitor, RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model' % (epoch + 1, self.monitor, self.best, current))
                    self.best = current

                    self.encoder.save(self.encoder_path, overwrite=True)
                    self.decoder.save(self.decoder_path, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve from %0.5f' % (epoch + 1, self.monitor, self.best))


# Testing site
# from keras.models import Model
# from keras.layers import Input, Concatenate, Conv2D, Conv2DTranspose, Lambda
#
# shape = (28, 28, 1)
#
# inp = Input(shape=shape)
# enc1 = Conv2D(2, (2, 2), strides=2)(inp)
# enc2 = Conv2D(2, (4, 4), strides=2)(inp)
#
# encoder = Model(inputs=inp, outputs=[enc1, enc2])
#
# dec_inp1 = Input(shape=(14, 14, 2))
# dec_inp2 = Input(shape=(13, 13, 2))
#
# dec1 = Conv2DTranspose(2, (2, 2), strides=2)(dec_inp1)
# dec2 = Conv2DTranspose(2, (4, 4), strides=2)(dec_inp2)
#
# dec = Concatenate()([dec1, dec2])
# decout = Conv2D(1, (1, 1))(dec1)
#
# decoder = Model(inputs=[dec_inp1, dec_inp2], outputs=decout)
#
# ae = AutoencoderTraining(decoder, encoder)
# alpha = 1e-2
# m = ae.network_compile(alpha, optimizer='adam')
#
#
# m.summary()
#
# n = 10
# X = np.random.uniform(0, 1, (n, ) + shape)
# delta = np.random.choice([-1, 1], (n, 1))
# y = X
#
# m.fit([X, delta], [y, np.zeros(n)], epochs=10)
#
#
# y_pred = m.predict([X, delta])
