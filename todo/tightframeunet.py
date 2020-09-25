from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, BatchNormalization
from keras.regularizers import l2
import keras.initializers as initializers
import numpy as np
from keras.layers import DepthwiseConv2D, Concatenate
import keras.backend as K


def filters(ftype='LL', size=2, ic=1, oc=1, upsampling=False):
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


def SequentialLayer(ch, beta=0, ind='0'):
    conv = Conv2D(ch, (3, 3), padding='same', kernel_regularizer=l2(beta), name='SeqConvDown' + ind)
    bn = BatchNormalization(name='SeqBNDown' + ind)
    activation = Activation('relu', name='SeqActDown' + ind)
    layers = [conv, bn, activation]

    def layer(inp):
        out = inp
        for lay in layers:
            out = lay(out)
        return out

    return layer


def WaveletDecomposition(ch, ind=0):
    wave = []
    for typ in ['HH', 'HL', 'LH', 'LL']:
        fil = filters(typ, 2, ch, 1)
        name = ''.join(['WaveletDecomp', typ, str(ind)])
        lay = DepthwiseConv2D(kernel_size=(2, 2), strides=(2, 2), depthwise_initializer=fil, name=name, trainable=False,
                              use_bias=False, depth_multiplier=1)

        wave.append(lay)

    def decomp(inp):

        out = inp
        output = []

        for lay in wave:
            output.append(lay(out))

        return output

    return decomp


def WaveletComposition(ch, ind=0):
    wave = []
    for typ in ['HH', 'HL', 'LH', 'LL']:
        fil = filters(typ, 2, ch, ch, upsampling=True)
        name = ''.join(['WaveletConc', typ, str(ind)])
        lay = Conv2DTranspose(ch, kernel_size=(2, 2), strides=(2, 2), kernel_initializer=fil, name=name,
                              trainable=False,
                              use_bias=False)
        wave.append(lay)

    def comp(InputLayer):
        upsample = []
        for WaveInp, WaveLay in zip(InputLayer, wave):
            upsample.append(WaveLay(WaveInp))
        return upsample

    return comp


def getModelX(input_shape=(512, 512, 1), beta=10**(-4), gamma=10**(-4)):

    inp = Input(shape=input_shape)

    # DOWNSAMPLING
    seq11 = SequentialLayer(64, ind='1_1', beta=beta)(inp)
    seq12 = SequentialLayer(64, ind='1_2', beta=beta)(seq11)

    downsampling1 = WaveletDecomposition(64, ind=1)(seq12)

    seq21 = SequentialLayer(128, ind='2_1', beta=beta)(downsampling1[-1])
    seq22 = SequentialLayer(128, ind='2_2', beta=beta)(seq21)

    downsampling2 = WaveletDecomposition(128, ind=2)(seq22)

    seq31 = SequentialLayer(256, ind='3_1', beta=beta)(downsampling2[-1])
    seq32 = SequentialLayer(256, ind='3_2', beta=beta)(seq31)

    downsampling3 = WaveletDecomposition(256, ind=3)(seq32)

    seq41 = SequentialLayer(512, ind='4_1', beta=beta)(downsampling3[-1])
    seq42 = SequentialLayer(512, ind='4_2', beta=beta)(seq41)

    downsampling4 = WaveletDecomposition(512, ind=4)(seq42)

    seqLOW1 = SequentialLayer(1024, ind='LOW_1', beta=beta)(downsampling4[-1])
    seqLOW2 = SequentialLayer(512, ind='LOW_2', beta=beta)(seqLOW1)

    # UPSAMPLING
    upsampling4 = WaveletComposition(512, ind=4)(downsampling4[:-1] + [seqLOW2])
    conc4 = Concatenate()([seq42] + upsampling4)

    sequp41 = SequentialLayer(512, ind='Up4_1', beta=gamma)(conc4)
    sequp42 = SequentialLayer(256, ind='Up4_2', beta=gamma)(sequp41)

    upsampling3 = WaveletComposition(256, ind=3)(downsampling3[:-1] + [sequp42])
    conc3 = Concatenate()([seq32] + upsampling3)

    sequp31 = SequentialLayer(256, ind='Up3_1', beta=gamma)(conc3)
    sequp32 = SequentialLayer(128, ind='Up3_2', beta=gamma)(sequp31)

    upsampling2 = WaveletComposition(128, ind=2)(downsampling2[:-1] + [sequp32])
    conc2 = Concatenate()([seq22] + upsampling2)

    sequp21 = SequentialLayer(128, ind='Up2_1', beta=gamma)(conc2)
    sequp22 = SequentialLayer(64, ind='Up2_2', beta=gamma)(sequp21)

    upsampling1 = WaveletComposition(64, ind=1)(downsampling1[:-1] + [sequp22])
    conc1 = Concatenate()([seq12] + upsampling1)

    sequp11 = SequentialLayer(64, ind='Up1_1', beta=gamma)(conc1)
    sequp12 = SequentialLayer(64, ind='Up1_2', beta=gamma)(sequp11)

    out = Concatenate()([sequp12, inp])

    out = Conv2D(1, (1, 1))(out)
    return Model(inputs=inp, outputs=out)


