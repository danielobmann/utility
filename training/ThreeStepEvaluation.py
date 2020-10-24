import sys
sys.path.append('..')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Conv2D, BatchNormalization, PReLU, UpSampling2D, Add
import numpy as np
import tensorflow as tf
import odl
import odl.contrib.tensorflow
import matplotlib.pyplot as plt

sess = tf.Session()

# ---------------------------
# Specify parameters
epochs = 51
batch_size = 4
n_training_samples = 1709
n_validation_samples = 458
n_batches = n_training_samples//batch_size
n_batches_val = n_validation_samples//batch_size

initial_lr = 1e-3

size = 512
n_theta = 32
upsampling_factor = 23
n_s = 768

# ---------------------------
# Set up tomography operator

space = odl.uniform_discr([-128, -128], [128, 128], [size, size], dtype='float32', weighting=1.0)
angle_partition = odl.uniform_partition(0, np.pi, n_theta)
angle_partition_up = odl.uniform_partition(0, np.pi, n_theta*upsampling_factor)
detector_partition = odl.uniform_partition(-360, 360, n_s)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
geometry_up = odl.tomo.Parallel2dGeometry(angle_partition_up, detector_partition)

operator = odl.tomo.RayTransform(space, geometry)
pseudoinverse = odl.tomo.fbp_op(operator)

Radon = odl.tomo.RayTransform(space, geometry_up)
FBP = odl.tomo.fbp_op(Radon)

operator /= odl.operator.power_method_opnorm(operator)
Radon /= odl.operator.power_method_opnorm(Radon)

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator, 'RayTransform')
odl_op_layer_pseudo = odl.contrib.tensorflow.as_tensorflow_layer(pseudoinverse, 'RayTransformPseudo')

radon_layer = odl.contrib.tensorflow.as_tensorflow_layer(Radon, 'Radon')
radon_adjoint_layer = odl.contrib.tensorflow.as_tensorflow_layer(Radon.adjoint, 'RadonAdjoint')
fbp_layer = odl.contrib.tensorflow.as_tensorflow_layer(FBP, 'FBP')

# ---------------------------
# Set up denoising network
inp_shape = operator.range.shape + (1, )

inp = tf.placeholder(tf.float32, shape=(None,) + inp_shape, name='input_denoising')

out = Conv2D(64, (3, 3), padding='same')(inp)
out = BatchNormalization()(out)
out = PReLU()(out)

out = Conv2D(64, (3, 3), padding='same')(out)
out = BatchNormalization()(out)
out = PReLU()(out)

out = Conv2D(64, (3, 3), padding='same')(out)
out = BatchNormalization()(out)
out = PReLU()(out)

out = Conv2D(64, (3, 3), padding='same')(out)
out = BatchNormalization()(out)
out = PReLU()(out)

out = Conv2D(1, (3, 3), padding='same')(out)
out = PReLU()(out)

out = Add()([out, inp])

# Make output operator consistent
out = odl_op_layer_pseudo(out)
out = Conv2D(64, (10, 10), padding='same')(out)
out = BatchNormalization()(out)
out = PReLU()(out)
out = Conv2D(1, (1, 1), padding='same')(out)
out = odl_op_layer(out)
out = tf.identity(out, name='output_denoising')

# ---------------------------
# Define upsampling network
inp = tf.placeholder(tf.float32, shape=(None, ) + inp_shape, name='input_upsample')
out = UpSampling2D(size=(upsampling_factor, 1), interpolation='bilinear')(inp)

out = Conv2D(32, (3, 3), padding='same')(out)
out = BatchNormalization()(out)
out = PReLU()(out)

out = Conv2D(32, (3, 3), padding='same')(out)
out = BatchNormalization()(out)
out = PReLU()(out)

out = Conv2D(32, (3, 3), padding='same')(out)
out = BatchNormalization()(out)
out = PReLU()(out)

out = Conv2D(32, (3, 3), padding='same')(out)
out = BatchNormalization()(out)
out = PReLU()(out)

out = Conv2D(1, (3, 3), padding='same')(out)
out = PReLU()(out)

# Make output operator consistent
out = fbp_layer(out)
out = Conv2D(64, (10, 10), padding='same')(out)
out = BatchNormalization()(out)
out = PReLU()(out)
out = Conv2D(1, (1, 1), padding='same')(out)
out = radon_layer(out)
out = tf.identity(out, name='output_upsample')

# ---------------------------
# Define inversion network

n_primal, n_dual, n_iter = 5, 5, 5

input_x = tf.placeholder(tf.float32, shape=(None, size, size, 1), name='input_x')
input_y = tf.placeholder(tf.float32, shape=(None, n_theta*upsampling_factor, n_s, 1), name='input_y')


def apply_conv(inputs, filters=32):
    outputs = Conv2D(filters, kernel_size=(3, 3), padding='same')(inputs)
    outputs = PReLU()(outputs)
    return outputs


with tf.name_scope('tomography'):
    with tf.name_scope('initial_values'):
        primal = tf.concat([tf.zeros_like(input_x)] * n_primal, axis=-1)
        dual = tf.concat([tf.zeros_like(input_y)] * n_dual, axis=-1)

    for i in range(n_iter):
        with tf.variable_scope('dual_iterate_{}'.format(i)):
            evalpt = primal[..., 1:2]
            evalop = radon_layer(evalpt)
            update = tf.concat([dual, evalop, input_y], axis=-1)

            update = apply_conv(update)
            update = apply_conv(update)
            update = apply_conv(update, filters=n_dual)
            dual = dual + update

        with tf.variable_scope('primal_iterate_{}'.format(i)):
            evalpt_fwd = primal[..., 0:1]
            evalop_fwd = radon_layer(evalpt_fwd)

            evalpt = dual[..., 0:1]
            evalop = radon_adjoint_layer(evalop_fwd * evalpt)
            update = tf.concat([primal, evalop], axis=-1)

            update = apply_conv(update)
            update = apply_conv(update)
            update = apply_conv(update, filters=n_primal)
            primal = primal + update

    x_result = primal[..., 0:1]

output = tf.identity(x_result, name='output_inversion')
x_true = tf.placeholder(shape=(None, size, size, 1), dtype=tf.float32)
# ---------------------------
# Set up loss function for training
loss = tf.reduce_sum(tf.squared_difference(output, x_true))

learning_rate = tf.placeholder(dtype=tf.float32)
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = opt.minimize(loss)


sess.run(tf.global_variables_initializer())

# Restore graph from trained model
restore_path = "models/"
if 1:
    new_saver = tf.train.import_meta_graph(restore_path + 'inversion_network-0.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(restore_path))

graph = tf.get_default_graph()


# ---------------------------
# Define inputs and outputs of denoising/upsampling network
inp_denois = tf.get_default_graph().get_tensor_by_name("input_denoising:0")
out_denois = tf.get_default_graph().get_tensor_by_name("output_denoising:0")

inp_up = tf.get_default_graph().get_tensor_by_name("input_upsample:0")
out_up = tf.get_default_graph().get_tensor_by_name("output_upsample:0")

inp_inversion = tf.get_default_graph().get_tensor_by_name("input_y:0")
out_inversion = tf.get_default_graph().get_tensor_by_name("output_inversion:0")


# ---------------------------
# Define generator for data




def data_generator_inversion(batch_size=32, mode='train', rescale=1000.):
    p = data_path + mode
    files = np.random.choice(os.listdir(p), size=batch_size, replace=False)
    X = [np.load(p + '/' + file) for file in files]
    x_t = np.stack(X)[..., None]

    # Get low resolution sinograms
    y_n = [operator(x / rescale) for x in X]
    y_n = [y + np.random.normal(0, 1, y.shape) * sigma for y in y_n]
    y_n = np.stack(y_n)[..., None]

    return y_n, x_t


# ---------------------------
# Define generator for data
path = "../data/mayoclinic/data/full3mm/test"
sigma = 0.2
rescale = 1000.

for file in os.listdir(path):
    x = np.load(path + '/' + file)
    y_n = operator(x/rescale)
    y_n = y_n[None, ..., None]

    y_d = sess.run(out_denois, feed_dict={inp_denois: y_n})
    y_u = sess.run(out_up, feed_dict={inp_up: y_d})
    x_rec = sess.run(out_inversion, feed_dict={inp_inversion: y_u})

    plt.subplot(221)
    plt.imshow(x, cmap='bone')
    plt.axis('off')
    plt.title('True')

    plt.subplot(222)
    plt.imshow(pseudoinverse(y_n), cmap='bone')
    plt.axis('off')
    plt.title('FBP')

    plt.subplot(223)
    plt.imshow(x_rec[0, ..., 0], cmap='bone')
    plt.axis('off')
    plt.title('Prediction')

    plt.subplot(224)
    plt.imshow(np.abs(x - x_rec[0, ..., 0]), cmap='bone')
    plt.axis('off')
    plt.title('FBP')

    plt.savefig("images/test/" + file[-8:] + '.pdf', format='pdf')
    plt.clf()

