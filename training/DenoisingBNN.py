import sys
sys.path.append('..')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from imports.bayesianlayers import *
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import tensorflow as tf
import odl
import odl.contrib.tensorflow
import matplotlib.pyplot as plt

sess = tf.Session()

# ---------------------------
# Specify parameters
epochs = 1001
batch_size = 16
n_training_samples = 1709
n_validation_samples = 458
n_batches = n_training_samples//batch_size
n_batches_val = n_validation_samples//batch_size

initial_lr = 1e-2
lam = 1e-3

size = 512
n_theta = 32
n_s = 768

# Define the number of Monte-Carlo samples for approximating the datafit term
N_mc = 1

# ---------------------------
# Set up tomography operator

space = odl.uniform_discr([-128, -128], [128, 128], [size, size], dtype='float32', weighting=1.0)
angle_partition = odl.uniform_partition(0, np.pi, n_theta)
detector_partition = odl.uniform_partition(-360, 360, n_s)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

operator = odl.tomo.RayTransform(space, geometry)
pseudoinverse = odl.tomo.fbp_op(operator)

pseudoinverse *= odl.operator.power_method_opnorm(operator)
operator /= odl.operator.power_method_opnorm(operator)


# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator, 'RayTransform')
odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(operator.adjoint, 'RayTransformAdjoint')
odl_op_layer_pseudo = odl.contrib.tensorflow.as_tensorflow_layer(pseudoinverse, 'RayTransformAdjoint')

inp_shape = operator.range.shape + (1, )

# ---------------------------
# Define network architecture
# TODO: better denoising network architecture
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
y_true = tf.placeholder(shape=(None,) + inp_shape, dtype=tf.float32)

# ---------------------------
# Set up loss function for training
kl = denoiser.loss/N_mc
likelihood = tf.reduce_sum(tf.squared_difference(outputs, y_true))/(N_mc*batch_size)
kl_parameter = tf.placeholder(dtype=tf.float32)

loss = lam*kl*kl_parameter + likelihood

learning_rate = tf.placeholder(dtype=tf.float32)
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = opt.minimize(loss)

# ---------------------------
# Set up various functions


def plot_validation(y_in, y_pred, y_true, epoch=10):
    n_images = y_in.shape[0]
    for i in range(n_images):
        fig, axs = plt.subplots(nrows=2, ncols=2)
        im = axs[0, 0].imshow(y_true[i, ..., 0], cmap='bone')
        axs[0, 0].set_aspect(n_s / n_theta)
        axs[0, 0].axis('off')
        axs[0, 0].set_title('True')
        fig.colorbar(im, ax=axs[0, 0])

        im = axs[0, 1].imshow(y_in[i, ..., 0], cmap='bone')
        axs[0, 1].set_aspect(n_s / n_theta)
        axs[0, 1].axis('off')
        axs[0, 1].set_title('Input')
        fig.colorbar(im, ax=axs[0, 1])

        im = axs[1, 0].imshow(np.mean(y_pred[i, ...], axis=-1), cmap='bone')
        axs[1, 0].set_aspect(n_s / n_theta)
        axs[1, 0].axis('off')
        axs[1, 0].set_title('Prediction')
        fig.colorbar(im, ax=axs[1, 0])

        im = axs[1, 1].imshow(np.abs(y_true[i, ..., 0] - np.mean(y_pred[i, ...], axis=-1)), cmap='bone')
        axs[1, 1].set_aspect(n_s / n_theta)
        axs[1, 1].axis('off')
        axs[1, 1].set_title('Difference')
        fig.colorbar(im, ax=axs[1, 1])

        fig.savefig('images/ValidationImage_Epoch' + str(epoch) + '_' + str(i) + '.pdf', format='pdf')
        fig.clf()

        plt.subplot(221)
        plt.imshow(pseudoinverse(y_true[i, ..., 0]), cmap='bone')
        plt.axis('off')

        plt.subplot(222)
        plt.imshow(pseudoinverse(y_in[i, ..., 0]), cmap='bone')
        plt.axis('off')

        rec = np.stack([pseudoinverse(y_pred[i,...,k]) for k in range(N_mc)], axis=-1)

        plt.subplot(223)
        plt.imshow(np.mean(rec, axis=-1), cmap='bone')
        plt.axis('off')

        plt.subplot(224)
        plt.imshow(np.std(rec, axis=-1), cmap='bone')
        plt.colorbar()
        plt.axis('off')

        plt.savefig('images/ValidationImageReconstruction_Epoch' + str(epoch) + '_' + str(i) + '.pdf', format='pdf')
        plt.clf()
    pass


def cosine_decay(epoch, total, initial=1e-3):
    return initial/2.*(1 + np.cos(np.pi*epoch/total))


def kl_par(batch):
    return 2.**(- batch)


def NMSE(y_true, y_pred, N_mc):
    res = tf.squared_difference(y_true, y_pred)
    res = tf.reduce_sum(res, axis=[1, 2, 3])
    nom = tf.reduce_sum(y_true**2, axis=[1, 2, 3])
    return tf.reduce_mean(res/nom)/N_mc


def SSIM(y_true, y_pred, N_mc):
    axis = [1, 2, 3]
    mu_x = tf.reduce_mean(y_true, axis=axis)
    mu_y = tf.reduce_mean(y_pred, axis=axis)
    cov = tf.reduce_sum((y_true - mu_x)*(y_pred - mu_y), axis=axis)
    var_x = tf.reduce_mean((y_true - mu_x)**2, axis=axis)
    var_y = tf.reduce_mean((y_pred - mu_y)**2, axis=axis)
    denom = (2*mu_x*mu_y + 0.01)*(2*cov + 0.01)
    nom = (mu_x**2 + mu_y**2 + 0.01)*(var_x + var_y + 0.01)
    return tf.reduce_mean(denom/nom)/N_mc


nmse = NMSE(y_true, outputs, N_mc=N_mc)
ssim = SSIM(y_true, outputs, N_mc=N_mc)


# ---------------------------
# Set up data generator
data_path = "../data/mayoclinic/data/full3mm/"
sigma = 0.2


def data_generator(batch_size=32, mode='train', rescale=1000.):
    p = data_path + mode
    files = np.random.choice(os.listdir(p), size=batch_size, replace=False)
    X = [np.load(p + '/' + file) for file in files]
    y_true = [operator(x/rescale) for x in X]
    y_noisy = [y + np.random.normal(0, 1, y.shape)*sigma for y in y_true]
    y_true = np.stack(y_true)[..., None]
    y_noisy = np.stack(y_noisy)[..., None]
    return y_noisy, y_true


# ---------------------------
# Custom training loop

print("Initialization successful. Starting training...", flush=True)

sess.run(tf.global_variables_initializer())
hist = {'loss': [], 'nmse': [], 'loss_val': [], 'nmse_val': [], 'ssim': [], 'ssim_val': []}
saver = tf.train.Saver()

save_path = "models/denoising_network"
n_save = 50
n_plot = 50
n_val = 1

# Training loop
for i in range(epochs):
    ERR = []
    NMSE = []
    SSIM_ = []

    print("### Epoch %d/%d ###" % (i + 1, epochs))
    for j in range(n_batches):
        y_input, y_output = data_generator(batch_size=batch_size, mode='train')

        fd = {inp: y_input,
              y_true: y_output,
              learning_rate: cosine_decay(i, epochs, initial=initial_lr),
              kl_parameter: kl_par(j)}

        c, nm, sm, _ = sess.run([loss, nmse, ssim, train_op], feed_dict=fd)
        NMSE.append(nm)
        ERR.append(c)
        SSIM_.append(sm)

    print("   Training: Loss %f NMSE %f SSIM %f" % (np.mean(ERR), np.mean(NMSE), np.mean(SSIM_)), end='\r', flush=True)
    hist['loss'].append(np.mean(ERR))
    hist['nmse'].append(np.mean(NMSE))
    hist['ssim'].append(np.mean(SSIM_))


    # Validate model performance
    if i % n_val == 0:
        ERR_VAL = []
        NMSE_VAL = []
        SSIM_VAL = []
        for j in range(n_batches_val):
            y_input, y_output = data_generator(batch_size=batch_size, mode='val')

            fd = {inp: y_input,
                  y_true: y_output,
                  kl_parameter: 1.0}

            c, nm, sm = sess.run([loss, nmse, ssim], feed_dict=fd)
            ERR_VAL.append(c)
            NMSE_VAL.append(nm)
            SSIM_VAL.append(sm)
        print(" ")
        print("   Validation: Loss %f Validation NMSE %f SSIM %F" % (np.mean(ERR_VAL), np.mean(NMSE_VAL), np.mean(SSIM_VAL)))
        print(" ", flush=True)
        hist['loss_val'].append(np.mean(ERR_VAL))
        hist['nmse_val'].append(np.mean(NMSE_VAL))
        hist['ssim_val'].append(np.mean(SSIM_VAL))

    if (i % n_plot) == 0:
        y_input, y_output = data_generator(batch_size=batch_size, mode='val')

        fd = {inp: y_input,
              y_true: y_output}

        y_pred = sess.run(outputs, feed_dict=fd)
        plot_validation(y_input, y_pred, y_output, epoch=i)

    # Save model every
    if (i % n_save) == 0:
        saver.save(sess, save_path, global_step=i)


plt.semilogy(hist['loss'])
plt.semilogy(hist['loss_val'])
plt.savefig('images/loss.pdf', format='pdf')

np.save("models/hist.npy", hist)
