import sys
sys.path.append('..')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from imports.networks import *
from imports.gan_training import *

latent_dim = 200
img_height, img_width, img_channels = 512, 512, 1

data_path = open('datapath').readline()
img_path = 'images/'

nets = Networks(img_height=img_height, img_width=img_width, img_channels=img_channels)

generator = nets.get_generator(latent_dim=latent_dim)
discriminator = nets.get_discriminator()

generator.summary()
discriminator.summary()

epochs = 200
batch_size = 16
img_interval = 10
save_interval = 10

gan = GANTraining(generator, discriminator, data_path)
gan.train(epochs=epochs, batch_size=batch_size, save_interval=save_interval, img_interval=img_interval)

