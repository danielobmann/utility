import os
from keras.models import Model
from keras.layers import Input, Lambda
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
    l1-loss is scaled like 1/(64^2).'''

    @staticmethod
    def _tf_norm(x, p=2):
        x = tf.cast(x, dtype=tf.float32)
        x = tf.abs(x)**p
        # By choosing to only take the mean over the first axis we only take the mean over the batch.
        # This is done to avoid any unwanted scaling of the loss function if the encoded output has a different
        # dimension than the input/output of the autoencoder.
        return (1./p)*tf.reduce_sum(tf.reduce_mean(x, axis=0))

    def _l2_loss(self, y_true, y_pred):
        return self._tf_norm(y_true - y_pred, p=2)

    def _l1_loss(self, alpha):
        def loss(y_true, y_pred):
            return alpha*self._tf_norm(y_pred, p=1)
        return loss

    def network_compile(self, alpha, metrics=[], **kwargs):
        # Set up network for training and pass input through network for calculating the loss
        inp = self.encoder.inputs
        out_encoder = self.encoder(inp)
        out_decoder = self.decoder(out_encoder)

        # Delta controls whether the l1-regularization is applied or not.
        # In the case that the input is noise free, we choose delta = 1 and delta = 0 otherwise.
        # This ensures that only the noise-free images are mapped to something sparse!
        inp_delta = Input(shape=(1,))
        modify_encoded = Lambda(lambda x: [e * inp_delta[..., None, None] for e in x] if isinstance(x, list) else x*inp_delta[..., None, None],
                                name='modified_encoder_output')(out_encoder)

        output = [out_decoder] + (list(modify_encoded) if isinstance(modify_encoded, list) else [modify_encoded])
        network_train = Model(inputs=inp + [inp_delta], outputs=output)

        # Compile network
        losses = {'decoder_output': self._l2_loss,
                  'modified_encoder_output': self._l1_loss(alpha)}

        network_train.compile(**kwargs, loss=losses, metrics={'decoder_output': metrics})

        # Return network
        return network_train

    def get_data_generator_noise(self, batch_size, path, mode='train', sigma=0.05):
        p = path + '/' + mode
        while True:
            files = np.random.choice(os.listdir(p), size=batch_size, replace=True)
            imgs = [np.asarray(Image.open(p + '/' + file).convert('L')) / 255. for file in files]

            img_inputs = []
            delta_inputs = []

            for i in imgs:
                delta_inputs += [np.random.choice(2)]
                img_inputs += [i if delta_inputs[-1] == 1 else i + np.mean(i)*np.random.normal(0, 1, i.shape)*sigma]

            img_labels = np.stack(imgs)[..., None]
            img_inputs = np.stack(img_inputs)[..., None]
            delta_inputs = np.stack(delta_inputs)

            input = [img_inputs, delta_inputs]
            output = [img_labels] + [np.zeros(batch_size) for z in self.encoder.outputs]

            yield (input, output)


class AutoencoderCP(Callback):

    def __init__(self, filepath, encoder, decoder, encoder_path, decoder_path, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False, mode='auto', period=1):
        super(AutoencoderCP, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.encoder = encoder
        self.decoder = decoder

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
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
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            #self.model.save(filepath, overwrite=True)
                            self.encoder.save(self.encoder_path, overwrite=True)
                            self.decoder.save(self.decoder_path, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
                    self.encoder.save(self.encoder_path, overwrite=True)
                    self.decoder.save(self.decoder_path, overwrite=True)

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
# n = 10
# X = np.random.uniform(0, 1, (n, ) + shape)
# delta = np.random.choice(2, n)
# y = [X] + [np.zeros(n), np.zeros(n)]
#
# m.fit([X, delta], y, epochs=10)
# y_pred = m.predict([X, delta])
#
# Xp = m.predict([X, delta])[0]
# np.mean((Xp - X)**2)
#
# import keras.backend as K
#
# K.get_value(tf.reduce_sum(tf.reduce_mean(tf.squared_difference(Xp, X), axis=0)))
# np.sum(np.mean(((Xp - X)**2), axis=0))
#
# X.shape