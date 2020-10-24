import sys
sys.path.append('..')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from imports.datanetwork import *
from imports.forwardoperators import *
from imports.bayesianlayers import *
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt

sess = tf.Session()

# ----------------------------------------------------

size = 512
upsample_factor = 23
n_theta_small = 32
n_theta = n_theta_small*upsample_factor
n_s = 768


# Set up tomography operator
space = odl.uniform_discr([-128, -128], [128, 128], [size, size], dtype='float32', weighting=1.0)
angle_partition = odl.uniform_partition(0, np.pi, n_theta_small)
angle_partition_large = odl.uniform_partition(0, np.pi, n_theta)
detector_partition = odl.uniform_partition(-360, 360, n_s)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
geometry_large = odl.tomo.Parallel2dGeometry(angle_partition_large, detector_partition)

operator = odl.tomo.RayTransform(space, geometry)
pseudoinverse = odl.tomo.fbp_op(operator)

Radon = odl.tomo.RayTransform(space, geometry_large)
FBP = odl.tomo.fbp_op(Radon)

pseudoinverse *= odl.operator.power_method_opnorm(operator)
operator /= odl.operator.power_method_opnorm(operator)

# FBP = odl.operator.power_method_opnorm(Radon)
# Radon = odl.operator.power_method_opnorm(Radon)

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator, 'RayTransform')
odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(operator.adjoint, 'RayTransformAdjoint')
odl_op_layer_pseudo = odl.contrib.tensorflow.as_tensorflow_layer(pseudoinverse, 'RayTransformAdjoint')

# -------------------
# Define denoiser
N_mc = 1
inp_shape = operator.range.shape + (1, )
inp = tf.placeholder(shape=(None,)+inp_shape, dtype=tf.float32, name='input')

architecture = [BayesianConv2D(64, (3, 3), padding='same'),
                BatchNormalization(),
                BayesianPReLU(),
                BayesianConv2D(64, (3, 3), padding='same'),
                BatchNormalization(),
                BayesianPReLU(),
                BayesianConv2D(64, (3, 3), padding='same'),
                BatchNormalization(),
                BayesianPReLU(),
                BayesianConv2D(1, (3, 3), padding='same'),
                odl_op_layer_pseudo,
                BayesianConv2D(32, (10, 10), padding='same'),
                BatchNormalization(),
                BayesianPReLU(),
                BayesianConv2D(1, (1, 1), padding='same'),
                odl_op_layer]

denoiser = BayesianModel(architecture=architecture)

outputs = tf.concat([denoiser(inp) for i in range(N_mc)], axis=-1, name='output')


# -------------------
# Set up network and traininloop

DCS = DataConsistentNetwork(operator=Radon, pseudoinverse=FBP)
inp_dcs, out_dcs = DCS.network(inp_shape=(n_theta_small, n_s), steps=3)

y_true = tf.placeholder(tf.float32, shape=(None, n_theta, n_s, 1))
loss = tf.reduce_mean(tf.squared_difference(out_dcs, y_true))

learning_rate = tf.placeholder(dtype=tf.float32)
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = opt.minimize(loss)

sess.run(tf.global_variables_initializer())

# Restore graph from trained model
restore_path = "models/fbp_denoising/"
if 1:
    new_saver = tf.train.import_meta_graph(restore_path + 'denoising_network-1000.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(restore_path))

graph = tf.get_default_graph()


# -----------------------------
# Set up data generator
data_path = "../data/mayoclinic/data/full3mm/"
sigma = 0.2


def data_generator(batch_size=32, mode='train', rescale=1000.):
    p = data_path + mode
    files = np.random.choice(os.listdir(p), size=batch_size, replace=False)
    X = [np.load(p + '/' + file) for file in files]

    # Get high resolution sinograms
    y_t = [Radon(x/rescale) for x in X]
    y_t = np.stack(y_t)[..., None]

    # Get low resolution sinograms
    y_n = [operator(x/rescale) for x in X]
    y_n = [y + np.random.normal(0, 1, y.shape)*sigma for y in y_n]
    y_n = np.stack(y_n)[..., None]

    return y_n, y_t


def cosine_decay(epoch, total, initial=1e-3):
    return initial/2.*(1 + np.cos(np.pi*epoch/total))


# -----------------------------
# Set up loss function for training
save_path = "models/upsampling_network"

n_save = 10
n_plot = 1
n_val = 10

initial_lr = 1e-3
epochs = 51
batch_size = 2
n_training_samples = 1709
n_validation_samples = 458
n_batches = n_training_samples//batch_size + 1
n_batches_val = n_validation_samples//batch_size + 1

ERR = []
ERR_val = []

for epoch in range(epochs):
    print("############")
    print("Epoch %d " % (epoch+1))
    err = []
    err_val = []
    for batch in range(n_batches):
        # print("Progress %f " % ((batch+1)/n_batches), end='\r', flush=True)
        y_inp, y_true_ = data_generator(batch_size=batch_size, mode='train')
        y_denoised = sess.run(outputs, feed_dict={inp: y_inp})

        fd = {y_true: y_true_,
              inp_dcs: y_denoised,
              learning_rate: cosine_decay(epoch, epochs, initial=initial_lr)}

        l, _ = sess.run([loss, train_op], feed_dict=fd)
        err.append(l)

    ERR.append(np.mean(err))
    print("Train error %f " % (ERR[-1]))
    if (epoch+1) % n_val == 0:
        for batch in range(n_batches_val):
            y_inp, y_true_ = data_generator(batch_size=16, mode='val')
            y_denoised = sess.run(outputs, feed_dict={inp: y_inp})

            fd = {y_true: y_true_,
                  inp_dcs: y_denoised}

            l = sess.run(loss, feed_dict=fd)
            err_val.append(l)

        ERR_val.append(np.mean(err_val))
        print("Validation error %f " % (ERR_val[-1]), flush=True)

    if (epoch % n_save) == 0:
        new_saver.save(sess, save_path, global_step=epoch)

    if (epoch % n_plot) == 0:

        y_inp, y_true_ = data_generator(batch_size=batch_size, mode='val')
        y_denoised = sess.run(outputs, feed_dict={inp: y_inp})

        fd = {y_true: y_true_,
              inp_dcs: y_denoised}

        out = sess.run(out_dcs, feed_dict=fd)

        for i in range(batch_size):

            plt.subplot(221)
            plt.imshow(y_inp[i, ..., 0], cmap='bone')
            plt.axis('off')

            plt.subplot(222)
            plt.imshow(y_denoised[i, ..., 0], cmap='bone')
            plt.axis('off')

            plt.subplot(223)
            plt.imshow(y_true_[i, ..., 0], cmap='bone')
            plt.axis('off')

            plt.subplot(224)
            plt.imshow(out[i, ..., 0], cmap='bone')
            plt.axis('off')

            plt.savefig('images/dcs_validation_' + str(i) + '_' + str(epoch) + '.pdf', format='pdf')
            plt.clf()

            plt.subplot(221)
            plt.imshow(FBP(y_true_[i, ..., 0]), cmap='bone')
            plt.axis('off')

            plt.subplot(222)
            plt.imshow(pseudoinverse(y_inp[i, ..., 0]), cmap='bone')
            plt.axis('off')

            plt.subplot(223)
            plt.imshow(FBP(out[i, ..., 0]), cmap='bone')
            plt.axis('off')

            plt.subplot(224)
            plt.imshow(FBP(out[i, ..., 0] - y_true_[i, ..., 0]), cmap='bone')
            plt.axis('off')
            plt.colorbar()

            plt.savefig('images/dcs_validation_rec_' + str(i) + '_' + str(epoch) + '.pdf', format='pdf')
            plt.clf()


plt.semilogy(ERR)
plt.semilogy(ERR_val)
plt.savefig("images/dcs_loss.pdf", format='pdf')
