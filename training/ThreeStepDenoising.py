import sys
sys.path.append('..')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Conv2D, BatchNormalization, PReLU, Add
import numpy as np
import tensorflow as tf
import odl
import odl.contrib.tensorflow
import matplotlib.pyplot as plt

sess = tf.Session()

# ---------------------------
# Specify parameters
epochs = 21
batch_size = 8
n_training_samples = 1709
n_validation_samples = 458
n_batches = n_training_samples//batch_size
n_batches_val = n_validation_samples//batch_size

initial_lr = 1e-3

size = 512
n_theta = 32
n_s = 768

# ---------------------------
# Set up tomography operator

space = odl.uniform_discr([-128, -128], [128, 128], [size, size], dtype='float32', weighting=1.0)
angle_partition = odl.uniform_partition(0, np.pi, n_theta)
detector_partition = odl.uniform_partition(-360, 360, n_s)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

operator = odl.tomo.RayTransform(space, geometry)
pseudoinverse = odl.tomo.fbp_op(operator)

operator /= odl.operator.power_method_opnorm(operator)

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator, 'RayTransform')
odl_op_layer_pseudo = odl.contrib.tensorflow.as_tensorflow_layer(pseudoinverse, 'RayTransformPseudo')


# ---------------------------
# Define network architecture
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

y_true = tf.placeholder(shape=(None,) + inp_shape, dtype=tf.float32)


# ---------------------------
# Set up loss function for training
loss = tf.reduce_sum(tf.squared_difference(out, y_true))

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

        im = axs[1, 0].imshow(y_pred[i, ..., 0], cmap='bone')
        axs[1, 0].set_aspect(n_s / n_theta)
        axs[1, 0].axis('off')
        axs[1, 0].set_title('Prediction')
        fig.colorbar(im, ax=axs[1, 0])

        im = axs[1, 1].imshow(np.abs(y_true[i, ..., 0] - y_pred[i, ..., 0]), cmap='bone')
        axs[1, 1].set_aspect(n_s / n_theta)
        axs[1, 1].axis('off')
        axs[1, 1].set_title('Difference')
        fig.colorbar(im, ax=axs[1, 1])

        fig.savefig('images/DenoisingValidationImage_Epoch' + str(epoch) + '_' + str(i) + '.pdf', format='pdf')
        fig.clf()
    pass


def cosine_decay(epoch, total, initial=1e-3):
    return initial/2.*(1 + np.cos(np.pi*epoch/total))


nmse = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - out), axis=[1, 2, 3])/tf.reduce_sum(y_true**2, axis=[1, 2, 3]))


# ---------------------------
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
save_path = "models/denoising_network"
n_save = 10
n_val = 1

print("Initialization successful. Starting training...", flush=True)


sess.run(tf.global_variables_initializer())
hist = {'loss': [], 'nmse': [], 'loss_val': [], 'nmse_val': []}
saver = tf.train.Saver()


for i in range(epochs):
    ERR = []
    NMSE = []

    print("### Epoch %d/%d ###" % (i + 1, epochs))
    for j in range(n_batches):
        print("Progress %f" % ((j+1)/n_batches), end='\r', flush=True)
        y_input, y_output = data_generator(batch_size=batch_size, mode='train')

        fd = {inp: y_input,
              y_true: y_output,
              learning_rate: cosine_decay(i, epochs)}

        c, nm, _ = sess.run([loss, nmse, train_op], feed_dict=fd)
        NMSE.append(nm)
        ERR.append(c)

    print("   Training: Loss %f NMSE %f" % (np.mean(ERR), np.mean(NMSE)), end='\r', flush=True)
    hist['loss'].append(np.mean(ERR))
    hist['nmse'].append(np.mean(NMSE))


    # Validate model performance
    if i % n_val == 0:
        ERR_VAL = []
        NMSE_VAL = []
        for j in range(n_batches_val):
            y_input, y_output = data_generator(batch_size=batch_size, mode='val')

            fd = {inp: y_input,
                  y_true: y_output}

            c, nm = sess.run([loss, nmse], feed_dict=fd)
            ERR_VAL.append(c)
            NMSE_VAL.append(nm)
        print(" ")
        print("   Validation: Loss %f Validation NMSE %f" % (np.mean(ERR_VAL), np.mean(NMSE_VAL)))
        print(" ", flush=True)
        hist['loss_val'].append(np.mean(ERR_VAL))
        hist['nmse_val'].append(np.mean(NMSE_VAL))

    if (i % 10) == 0:
        y_input, y_output = data_generator(batch_size=batch_size, mode='val')

        fd = {inp: y_input,
              y_true: y_output}

        y_pred = sess.run(out, feed_dict=fd)
        plot_validation(y_input, y_pred, y_output, epoch=i)

    # Save model every
    if (i % n_save) == 0:
        saver.save(sess, save_path, global_step=i)


plt.semilogy(hist['loss'])
plt.semilogy(hist['loss_val'])
plt.savefig('images/denoising_loss.pdf', format='pdf')
