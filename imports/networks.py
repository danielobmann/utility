from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, BatchNormalization, Subtract
from keras.layers import Flatten, Reshape, LeakyReLU, Dense, Dropout, SpatialDropout2D
from keras.regularizers import l2, l1
import keras.initializers as initializers
import numpy as np
from keras.layers import DepthwiseConv2D, Concatenate

import os
from PIL import Image


class Networks:
    def __init__(self, img_height=512, img_width=512, img_channels=1):
        self._img_height = img_height
        self._img_width = img_width
        self._img_channels = img_channels
        self._input_shape = (img_height, img_width, img_channels)

    @staticmethod
    def _haar_filters(ftype='LL', size=2, ic=1, oc=1, upsampling=False):
        d = {'L': np.array([1. for j in range(size)]), 'H': np.array([(-1.) ** (j + 1) for j in range(size)])}

        weights = np.outer(d[ftype[0]], d[ftype[1]]) / 2.
        out = np.zeros((size, size, ic, oc))

        for i in range(ic):

            if upsampling:
                out[:, :, i, i] = weights

            else:
                for j in range(oc):
                    out[:, :, i, j] = weights

        return initializers.Constant(value=out)

    def wavelet_decomposition(self, channels, alpha=0., filters='haar'):
        wave = []
        if filters == 'haar':
            for typ in ['HH', 'HL', 'LH']:
                fil = self._haar_filters(typ, 2, channels, 1)
                lay = DepthwiseConv2D(kernel_size=(2, 2),
                                      strides=(2, 2),
                                      depthwise_initializer=fil,
                                      trainable=False,
                                      use_bias=False,
                                      depth_multiplier=1,
                                      activity_regularizer=l1(alpha))

                wave.append(lay)
            fil = self._haar_filters('LL', size=2, ic=channels, oc=1)
            lay = DepthwiseConv2D(kernel_size=(2, 2),
                                  strides=(2, 2),
                                  depthwise_initializer=fil,
                                  trainable=False,
                                  use_bias=False,
                                  depth_multiplier=1)
            wave.append(lay)

        def decomp(inp):
            output = []
            for layer in wave:
                output.append(layer(inp))
            return output
        return decomp

    def wavelet_composition(self, channels, filters='haar'):
        wave = []
        if filters == 'haar':
            for typ in ['HH', 'HL', 'LH', 'LL']:
                fil = self._haar_filters(typ, 2, channels, channels, upsampling=True)
                lay = Conv2DTranspose(channels,
                                      kernel_size=(2, 2),
                                      strides=(2, 2),
                                      kernel_initializer=fil,
                                      trainable=False,
                                      use_bias=False)
                wave.append(lay)

        def comp(inputlayer):
            upsample = []
            for WaveInput, WaveLayer in zip(inputlayer, wave):
                upsample.append(WaveLayer(WaveInput))
            return Concatenate()(upsample)
        return comp

    @staticmethod
    def sequential_layer(channels, filtersize=(3, 3), beta=0., activation='relu', use_batch_normalization=True):
        conv = Conv2D(channels, filtersize, padding='same', kernel_regularizer=l2(beta))
        layers = [conv]
        if use_batch_normalization:
            layers.append(BatchNormalization())
        layers.append(Activation(activation))

        def layer(inp):
            out = inp
            for lay in layers:
                out = lay(out)
            return out

        return layer

    def _get_autoencoder(self, channels=64, channels_enc=2, filtersize=(3, 3), alpha=0., beta=0., activation='relu'):
        inp = Input(shape=self._input_shape)
        output = []

        seq11 = self.sequential_layer(channels=channels, filtersize=filtersize, beta=beta, activation=activation)(inp)
        seq12 = self.sequential_layer(channels=channels_enc, filtersize=filtersize, beta=beta, activation=activation)(seq11)

        downsampling1 = self.wavelet_decomposition(channels=channels_enc, alpha=alpha)(seq12)
        output += downsampling1[:3]

        seq21 = self.sequential_layer(channels=2*channels, filtersize=filtersize, beta=beta, activation=activation)(downsampling1[-1])
        seq22 = self.sequential_layer(channels=2*channels_enc, filtersize=filtersize, beta=beta, activation=activation)(seq21)

        downsampling2 = self.wavelet_decomposition(channels=channels_enc, alpha=alpha/2.)(seq22)
        output += downsampling2[:3]

        seq31 = self.sequential_layer(channels=4*channels, filtersize=filtersize, beta=beta, activation=activation)(downsampling2[-1])
        seq32 = self.sequential_layer(channels=4*channels_enc, filtersize=filtersize, beta=beta, activation=activation)(seq31)

        downsampling3 = self.wavelet_decomposition(channels=4*channels_enc, alpha=alpha/4.)(seq32)
        output += downsampling3[:3]

        seq41 = self.sequential_layer(channels=8*channels, filtersize=filtersize, beta=beta, activation=activation)(downsampling3[-1])
        seq42 = self.sequential_layer(channels=8*channels_enc, filtersize=filtersize, beta=beta, activation=activation)(seq41)

        downsampling4 = self.wavelet_decomposition(channels=8*channels_enc, alpha=alpha/8.)(seq42)
        output += downsampling4[:3]

        seqlow1 = self.sequential_layer(channels=16*channels, filtersize=filtersize, beta=beta, activation=activation)(downsampling4[-1])
        seqlow2 = Conv2D(channels_enc * 16, (3, 3), padding='same', kernel_regularizer=l2(beta),
                         activity_regularizer=l1(alpha / 8), activation=activation)(seqlow1)

        output += [seqlow2]

        encoder = Model(inputs=inp, outputs=output)
        decoder_inputs = [Input(shape=[t.value for t in s.shape[1:]]) for s in encoder.outputs]

        upsampling4 = self.wavelet_composition(channels=8*channels_enc)(decoder_inputs[-4:])

        sequp41 = self.sequential_layer(channels=8*channels, filtersize=filtersize, beta=beta, activation=activation)(upsampling4)
        sequp42 = self.sequential_layer(channels=8*channels, filtersize=filtersize, beta=beta, activation=activation)(sequp41)

        upsampling3 = self.wavelet_composition(channels=4*channels_enc)(decoder_inputs[-7:-4] + [sequp42])

        sequp31 = self.sequential_layer(channels=4*channels, filtersize=filtersize, beta=beta, activation=activation)(upsampling3)
        sequp32 = self.sequential_layer(channels=4*channels, filtersize=filtersize, beta=beta, activation=activation)(sequp31)

        upsampling2 = self.wavelet_composition(channels=2*channels_enc)(decoder_inputs[-10:-7] + [sequp32])

        sequp21 = self.sequential_layer(channels=2*channels, filtersize=filtersize, beta=beta, activation=activation)(upsampling2)
        sequp22 = self.sequential_layer(channels=2*channels, filtersize=filtersize, beta=beta, activation=activation)(sequp21)

        upsampling1 = self.wavelet_composition(channels=channels_enc)(decoder_inputs[-13:-10] + [sequp22])

        sequp11 = self.sequential_layer(channels=2*channels, filtersize=filtersize, beta=beta, activation=activation)(upsampling1)
        sequp12 = self.sequential_layer(channels=2*channels, filtersize=filtersize, beta=beta, activation=activation)(sequp11)

        out = Conv2D(1, (1, 1))(sequp12)
        decoder = Model(inputs=decoder_inputs, outputs=out)

        ae_out = decoder(encoder(inp))
        model = Model(inputs=inp, outputs=ae_out)
        return model, decoder, encoder

    def get_autoencoder(self, steps=4, channels=64, channels_enc=2, filtersize=(3, 3), alpha=0., beta=0., activation='relu'):
        inp = Input(shape=self._input_shape)
        output_loop = [inp]

        # Downsampling
        for step in range(steps):
            out = self.sequential_layer(channels=channels * 2 ** step, filtersize=filtersize, beta=beta,
                                        activation=activation)(output_loop[-1])
            out = self.sequential_layer(channels=channels_enc * 2 ** step, filtersize=filtersize, beta=beta,
                                        activation=activation)(out)

            out = self.wavelet_decomposition(channels=channels_enc*2**step, alpha=alpha*2**(-step))(out)
            output_loop += out

        # Lowest layer
        seqlow1_loop = self.sequential_layer(channels=channels*2**steps, filtersize=filtersize, beta=beta,
                                             activation=activation)(output_loop[-1])
        seqlow2_loop = Conv2D(channels_enc*2**steps, (3, 3), padding='same', kernel_regularizer=l2(beta), activation=activation)(seqlow1_loop)
        seqlow_out = Conv2D(channels_enc*2**steps, (3, 3), padding='same', kernel_regularizer=l2(beta),
                            activity_regularizer=l1(alpha / 8), activation=activation)(seqlow2_loop)

        output_loop += [seqlow_out]

        output = list(np.delete(output_loop[1:], slice(3, None, 4)))
        encoder = Model(inputs=inp, outputs=output)

        decoder_inputs = [Input(shape=[t.value for t in s.shape[1:]]) for s in encoder.outputs]
        out = self.wavelet_composition(channels=channels_enc*2**(steps-1))(decoder_inputs[-4:])

        for step in range(steps-1):
            k = (steps-step-1)

            out = self.sequential_layer(channels=channels*2**k, filtersize=filtersize, beta=beta, activation=activation)(out)
            out = self.sequential_layer(channels=channels*2**k, filtersize=filtersize, beta=beta, activation=activation)(out)
            out = self.wavelet_composition(channels=channels_enc*2**(k-1))(decoder_inputs[-(7+3*step):-(4+3*step)] + [out])

        out = self.sequential_layer(channels=2*channels, filtersize=filtersize, beta=beta, activation=activation)(out)
        out = self.sequential_layer(channels=2*channels, filtersize=filtersize, beta=beta, activation=activation)(out)
        out = Conv2D(1, (1, 1))(out)

        decoder = Model(inputs=decoder_inputs, outputs=out)
        ae_out = decoder(encoder(inp))
        model = Model(inputs=inp, outputs=ae_out)
        return model, decoder, encoder

    def get_residual_network(self, steps=4, channels=64, filtersize=(3, 3), beta=0., activation='relu'):
        inp = Input(shape=self._input_shape)
        output_loop = [inp]
        output_skip = [inp]

        # Downsampling
        for step in range(steps):
            out = self.sequential_layer(channels=channels * 2 ** step, filtersize=filtersize, beta=beta,
                                        activation=activation)(output_loop[-1])
            out = self.sequential_layer(channels=channels * 2 ** step, filtersize=filtersize, beta=beta,
                                        activation=activation)(out)
            output_skip.append(out)

            out = self.wavelet_decomposition(channels=channels * 2 ** step)(out)
            output_loop += out

        # Lowest layer
        seqlow1_loop = self.sequential_layer(channels=channels * 2 ** steps, filtersize=filtersize, beta=beta,
                                             activation=activation)(output_loop[-1])
        seqlow2_loop = Conv2D(channels * 2 ** steps, (3, 3), padding='same', kernel_regularizer=l2(beta),
                              activation=activation)(seqlow1_loop)
        seqlow_out = Conv2D(channels * 2 ** steps, (3, 3), padding='same', kernel_regularizer=l2(beta),
                            activation=activation)(seqlow2_loop)

        output_loop += [seqlow_out]

        output = list(np.delete(output_loop[1:], slice(3, None, 4)))
        out = self.wavelet_composition(channels=channels * 2 ** (steps - 1))(output[-4:])

        for step in range(steps - 1):
            k = (steps - step - 1)

            out = self.sequential_layer(channels=channels * 2 ** k, filtersize=filtersize, beta=beta,
                                        activation=activation)(out)
            out = self.sequential_layer(channels=channels * 2 ** k, filtersize=filtersize, beta=beta,
                                        activation=activation)(out)
            out = self.wavelet_composition(channels=channels * 2 ** (k - 1))(output[-(7 + 3 * step):-(4 + 3 * step)]+[out])

            out = Concatenate()([out] + [output_skip[k]])

        out = self.sequential_layer(channels=2 * channels, filtersize=filtersize, beta=beta, activation=activation)(out)
        out = self.sequential_layer(channels=2 * channels, filtersize=filtersize, beta=beta, activation=activation)(out)
        out = Conv2D(1, (1, 1))(out)
        out = Concatenate()([inp, out])
        out = Conv2D(1, (1, 1))(out)

        return Model(inputs=inp, outputs=out)

    def get_convolutional_autoencoder(self, steps=4, channels=64, filtersize=(3, 3), alpha=0., beta=0., activation='relu',
                                      latent_dim=512, p=0.0, use_batch_normalization=True):
        inp = Input(shape=self._input_shape)
        out = inp
        for step in range(steps):
            out = self.sequential_layer(channels=channels*2**step, filtersize=filtersize, beta=beta,
                                        activation=activation, use_batch_normalization=use_batch_normalization)(out)
            out = self.sequential_layer(channels=channels*2**step, filtersize=filtersize, beta=beta,
                                        activation=activation, use_batch_normalization=use_batch_normalization)(out)

            out = Conv2D(filters=channels*2**step, kernel_size=(2, 2), strides=2, activation=activation,
                         kernel_regularizer=l2(beta))(out)
            out = SpatialDropout2D(p)(out)

        out = Conv2D(latent_dim, kernel_size=(2, 2), padding='same', activation=activation,
                     activity_regularizer=l1(alpha))(out)
        temp_shape = [int(i) for i in out.shape[1:]]
        encoder = Model(inputs=inp, outputs=out)

        decoder_input = Input(shape=temp_shape)
        out = Reshape(temp_shape)(decoder_input)

        for step in range(steps):
            k = int(steps - step - 1)
            out = self.sequential_layer(channels=channels*2**k, filtersize=filtersize, beta=beta,
                                        activation=activation, use_batch_normalization=use_batch_normalization)(out)
            out = self.sequential_layer(channels=channels*2**k, filtersize=filtersize, beta=beta,
                                        activation=activation, use_batch_normalization=use_batch_normalization)(out)

            out = Conv2DTranspose(filters=channels*2**k, kernel_size=(2, 2), strides=2, activation=activation,
                                  kernel_regularizer=l2(beta))(out)
            out = SpatialDropout2D(p)(out)

        out = Conv2D(self._img_channels, (1, 1))(out)
        decoder = Model(inputs=decoder_input, outputs=out)
        ae_out = decoder(encoder(inp))
        ae = Model(inputs=inp, outputs=ae_out)
        return ae, decoder, encoder

    def get_u_net(self, steps=4, channels=64, filtersize=(3, 3), beta=0., activation='relu'):
        inp = Input(shape=self._input_shape)
        out = inp
        for step in range(steps):
            out = Conv2D(channels*2**step, filtersize, activation=activation, kernel_regularizer=l2(beta),
                         padding='same')(out)
        pass

    def get_generator(self, latent_dim=500, steps=4, channels=64, filtersize=(3, 3)):
        inp = Input(shape=(latent_dim,))
        K = int(np.log2(self._img_height) - steps)

        out = Dense(latent_dim)(inp)
        out = Dense(2**(K*2)*self._img_channels)(out)
        out = BatchNormalization()(out)
        out = LeakyReLU()(out)

        out = Reshape((2**K, 2**K, self._img_channels))(out)
        out = Conv2D(channels, (7, 7), padding='same')(out)
        out = BatchNormalization()(out)
        out = LeakyReLU()(out)

        for step in range(steps):

            out = Conv2D(channels * 2 ** step, filtersize, padding='same')(out)
            out = LeakyReLU()(out)
            out = Conv2D(channels * 2 ** step, filtersize, padding='same')(out)
            out = LeakyReLU()(out)
            out = Conv2DTranspose(channels * 2 ** step, (2, 2), strides=2)(out)
            out = BatchNormalization()(out)
            out = LeakyReLU()(out)

        out = Conv2D(1, (1, 1))(out)

        return Model(inp, out)

    def get_discriminator(self, steps=4, channels=64, filtersize=(4, 4)):
        inp = Input(shape=self._input_shape)
        out = inp

        for step in range(steps):
            out = Conv2D(channels * 2 ** step, filtersize, padding='same')(out)
            out = LeakyReLU()(out)
            out = Conv2D(channels * 2 ** step, (2, 2), strides=2)(out)
            out = BatchNormalization()(out)
            out = LeakyReLU()(out)

        out = Conv2D(channels, filtersize)(out)

        out = Flatten()(out)
        out = Dense(256)(out)
        out = LeakyReLU()(out)

        out = Dense(1)(out)
        out = Activation('sigmoid')(out)

        return Model(inp, out)

    def get_data_generator(self, batch_size, path, mode='train', processor_input=None, processor_label=None):
        p = path + '/' + mode
        while True:
            files = np.random.choice(os.listdir(p), size=batch_size, replace=True)
            imgs = [np.asarray(Image.open(p + '/' + file).convert('L')) / 255. for file in files]

            inputs = [processor_input(a) if processor_input is not None else a for a in imgs]
            labels = [processor_label(a) if processor_label is not None else a for a in imgs]

            inputs = np.concatenate([a.reshape((1, ) + self._input_shape) for a in inputs])
            labels = np.concatenate([a.reshape((1, ) + self._input_shape) for a in labels])
            yield (inputs, labels)

    def get_prediction(self, x, network):
        inp = x.reshape((1, ) + self._input_shape)
        if self._img_channels == 1:
            ret = network.predict(inp)[0, ..., 0]
        else:
            ret = network.predict(inp)[0, ...]
        return ret
