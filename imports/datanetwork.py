from tensorflow.keras.layers import Conv2D, PReLU, BatchNormalization, UpSampling2D, Concatenate
import tensorflow as tf
import odl.contrib.tensorflow
import numpy as np


class DataConsistentNetwork:

    def __init__(self, operator, pseudoinverse):
        self._operator = operator
        self._psuedoinverse = pseudoinverse
        self._operator_tensorflow = odl.contrib.tensorflow.as_tensorflow_layer(operator, 'Operator')
        self._pseudoinverse_tensorflow = odl.contrib.tensorflow.as_tensorflow_layer(pseudoinverse, 'Pseudo')

    def input_layer(self, inp_shape):
        out_shape = self._operator.range.shape
        # Return input layer with appropriate input size
        inp = tf.placeholder(tf.float32, shape=(None, ) + inp_shape + (1,), name='input_dcs')
        upsampling_factor = out_shape[0]//inp_shape[0]

        # Simple upsampling, e.g. by filling with zeros
        self._mask = np.zeros(out_shape)
        for i in range(inp_shape[0]):
            self._mask[i*upsampling_factor, ...] = 1

        self._mask = tf.constant(self._mask[None, ..., None], dtype=tf.float32, name='mask')
        out = UpSampling2D(size=(upsampling_factor, 1), interpolation='bilinear')(inp)
        y0 = out*self._mask
        return inp, out, y0

    @staticmethod
    def _convolution_block(out, global_step, steps=3, filters=32, kernel_size=(3, 3)):
        out = Conv2D(1, (1, 1), padding='same')(out)

        for step in range(steps):
            out = Conv2D(filters, kernel_size, padding='same')(out)
            out = BatchNormalization(name='batch_dcs_' + str(global_step) + '_' + str(step))(out)
            out = PReLU(shared_axes=[1, 2])(out)

        return out

    def _data_consistency_block(self, inp, out, y0, global_step, filters=32, kernel_size=(3, 3)):

        out = Concatenate()([inp, out, y0])
        out = self._convolution_block(out, global_step=global_step, filters=filters, kernel_size=kernel_size)
        out = Conv2D(1, (1, 1), padding='same')(out)

        # Enforce consistency with data
        out = out*(1-self._mask) + y0

        return inp, out, y0

    def _operator_consistency_block(self, inp, out, y0, global_step, filters=32, kernel_size=(3, 3)):

        out = Concatenate()([inp, out, y0])
        out = self._convolution_block(out, global_step=global_step, filters=filters, kernel_size=kernel_size)
        out = Conv2D(1, (1, 1), padding='same')(out)

        # Enforce operator consistency
        out = self._pseudoinverse_tensorflow(out)
        out = Conv2D(filters, (10, 10), padding='same')(out)
        out = BatchNormalization()(out)
        out = PReLU(shared_axes=[1, 2])(out)
        out = Conv2D(1, (1, 1))(out)
        out = self._operator_tensorflow(out)

        return inp, out, y0

    def network(self, inp_shape, steps=3, filters=32, kernel_size=(3, 3)):
        inp, out_upsample, y0 = self.input_layer(inp_shape)
        out = out_upsample
        # Go in steps of 2 to have a unique global step
        for i in range(0, steps, 2):
            out_upsample, out, y0 = self._data_consistency_block(out_upsample, out, y0, global_step=i,
                                                                 filters=filters, kernel_size=kernel_size)
            out_upsample, out, y0 = self._operator_consistency_block(out_upsample, out, y0, global_step=i+1,
                                                                     filters=filters, kernel_size=kernel_size)
        out = tf.identity(out, name='output_dcs')
        return inp, out


# # Testing site
# import odl
#
# size = 512
# n_theta = 23*32
# n_s = 768
#
# space = odl.uniform_discr([-1, -1], [1, 1], [size, size], dtype='float32', weighting=1.0)
# angle_partition = odl.uniform_partition(0, np.pi, n_theta)
# detector_partition = odl.uniform_partition(-1.5, 1.5, n_s)
# geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
#
# Radon = odl.tomo.RayTransform(space, geometry)
# FBP = odl.tomo.fbp_op(Radon)
#
# dcs = DataConsistentNetwork(Radon, FBP)
# inp_dcs, out_dcs = dcs.network((32, 768))
#
