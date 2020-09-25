from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, Dense, BatchNormalization
from keras.layers import Reshape, Flatten, LeakyReLU
from keras.regularizers import l2
import numpy as np
import keras.backend as K


def generator(input_dim=100, img_size=(512, 512)):
    inp = Input(shape=(input_dim, ))

    out = Dense(input_dim)(inp)
    out = Dense(np.prod(img_size))(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)

    out = Reshape(img_size + (1,))(out)
    out = Conv2D(128, (7, 7))(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)

    out = Conv2DTranspose(128, (7, 7))(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)

    out = Conv2D(128, (7, 7), padding='same')(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)

    out = Conv2D(128, (7, 7), padding='same')(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)

    out = Conv2D(1, (1,1))(out)

    return Model(inp, out)


def discriminator(img_size=(512, 512)):
    inp = Input(shape=img_size + (1,))

    out = Conv2D(128, (3, 3), strides=2)(inp)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)

    out = Conv2D(128, (4, 4), strides=2)(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)

    out = Conv2D(256, (4, 4), strides=2)(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)

    out = Conv2D(128, (4, 4), strides=2)(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)

    out = Conv2D(128, (4, 4), strides=2)(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)

    out = Flatten()(out)
    out = Dense(256)(out)
    out = LeakyReLU()(out)

    out = Dense(1)(out)
    out = Activation('sigmoid')(out)

    return Model(inp, out)


def generatorloss(discriminator):
    def loss(y_true, y_pred):
        return -K.log(discriminator(y_pred))
    return loss


def set_trainable(model, mode=True):
    for layer in model.layers:
        layer.trainable = mode
    pass
