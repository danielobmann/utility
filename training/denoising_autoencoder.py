import sys
sys.path.append('..')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from imports.customobjects import *
from imports.networks import *
from imports.ae_training import *

import keras.backend as K
from keras.models import load_model

import optparse

# ----------------------------------------------------
# Parameter parsing
parser = optparse.OptionParser()
parser.add_option('-a', '--alpha', action="store", dest="alpha", default=10**(-3))
parser.add_option('-b', '--beta', action="store", dest="beta", default=10**(-5))
parser.add_option('-c', '--channels', action="store", dest="channels", default=64)
parser.add_option('-e', '--channelsenc', action="store", dest="channelsenc", default=2)
parser.add_option('-l', '--latent', action="store", dest="latent", default=2048)
parser.add_option('-s', '--steps', action="store", dest="steps", default=4)
parser.add_option('-p', '--dropout', action="store", dest="dropout", default=0.0)
parser.add_option('-n', '--batchnormalization', action="store", dest="batchnormalization", default=True)
parser.add_option('-i', '--index', action="store", dest="index", default=0)

options, args = parser.parse_args()

alpha = float(options.alpha)
beta = float(options.beta)
channels = int(options.channels)
channels_enc = int(options.channelsenc)
latent_dim = int(options.latent)
steps = int(options.steps)
p = float(options.dropout)
batch_normalization = bool(options.batchnormalization)
save_index = int(options.index)

print("######## Parameter Log ########", flush=True)
print("Alpha", alpha, flush=True)
print("Beta", beta, flush=True)
print("Channels", channels, flush=True)
print("Channels encoded", channels_enc, flush=True)
print("Latent dimension", latent_dim, flush=True)
print("Downsampling steps", steps, flush=True)
print("Dropoutrate", p, flush=True)
print("Use BN", batch_normalization, flush=True)
print("Saving to index", save_index, flush=True)
print("###############################")


img_height, img_width, img_channels = 512, 512, 1
activation = 'relu'
#alpha /= (img_height*img_width)
#beta /= (img_height*img_width)

batch_size = 4
epochs = 100
spe = 4000 // batch_size
val_steps = 1000 // batch_size
sigma = 0.2

datapath = open('datapath').readline()
# ----------------------------------------------------

sess = K.get_session()
co = CustomObjects(sess)

nets = Networks(img_height=img_height, img_width=img_width, img_channels=img_channels)

#m, d, e = nets.get_autoencoder(channels=channels, channels_enc=channels_enc, alpha=0, beta=beta, activation=activation)
m, d, e = nets.get_convolutional_autoencoder(steps=steps, channels=channels, latent_dim=latent_dim, alpha=0, beta=beta, p=p, use_batch_normalization=batch_normalization)


# Set up the network for training
training = AutoencoderTraining(d, e)
training_network = training.network_compile(alpha=alpha, optimizer='adam', metrics=['mse', co.KerasPSNR, co.KerasNMSE])

t_generator = training.get_data_generator_noise(batch_size=batch_size, path=datapath, mode='train', sigma=sigma)
v_generator = training.get_data_generator_noise(batch_size=batch_size, path=datapath, mode='val', sigma=sigma)


batch = next(t_generator)
inputs = batch[0]
outputs = batch[1]

for i in range(batch_size):
    plt.subplot(121)
    plt.imshow(inputs[0][i, :, :, 0], cmap='gray')
    plt.title('Input')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(outputs[0][i, :, :, 0], cmap='gray')
    plt.title('Label')
    plt.axis('off')

    plt.savefig("images/GeneratorTest" + str(save_index) + "_Output" + str(i) + ".pdf", format='pdf')
    plt.clf()

model_save = "models/model" + str(save_index) + ".h5"
encoder_save = "models/encoder" + str(save_index) + ".h5"
decoder_save = "models/decoder" + str(save_index) + ".h5"


CP_autoencoder = AutoencoderCP(model_save, e, d, encoder_save, decoder_save, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
history_autoencoder = training_network.fit_generator(t_generator, steps_per_epoch=spe, epochs=epochs,
                                                     validation_data=v_generator, validation_steps=val_steps, verbose=2,
                                                     callbacks=[CP_autoencoder])

plt.semilogy(history_autoencoder.history['loss'], label='Trainloss')
plt.semilogy(history_autoencoder.history['val_loss'], label='Validationloss')
plt.legend()

plt.savefig('images/LossAutoencoder' + str(save_index) + '.pdf', format='pdf')
plt.clf()


# Plot some example images

def processor_input(x):
    m = np.mean(x)
    s = np.random.choice(np.linspace(0, sigma, 10))
    return x + np.random.normal(0, 1, x.shape)*s*m

m = load_model(model_save, custom_objects=co.custom_objects)

test_generator = nets.get_data_generator(batch_size=8, path=datapath, mode='test', processor_input=processor_input)

X = next(test_generator)
X_pred = m.predict_on_batch(X[0])

for i in range(batch_size):
    plt.subplot(131)
    plt.imshow(X[0][i, :, :, 0], cmap='gray')
    plt.title('Input')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(X_pred[i, :, :, 0], cmap='gray')
    plt.title('Model output')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(X[1][i, :, :, 0], cmap='gray')
    plt.title('Label')
    plt.axis('off')

    plt.savefig("images/Autoencoder" + str(save_index) + "_Output" + str(i) + ".pdf", format='pdf')
    plt.clf()
