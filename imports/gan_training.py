import numpy as np
import os
from PIL import Image
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import Adam


class GANTraining:

    def __init__(self, generator, discriminator, path, learning_rate=10**(-3), beta_1=0.5, beta_2=0.999):

        self._optimizer = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2)

        self.generator = generator
        self.generator.compile(loss='binary_crossentropy', optimizer=self._optimizer)

        self.discriminator = discriminator
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self._optimizer, metrics=['accuracy'])

        self.latent_dim = int(generator.inputs[0].shape[1:][0])
        self.img_size = tuple([int(i) for i in discriminator.inputs[0].shape[1:]])
        self.path = path

        # Define the combined model for training the generator
        z = Input(shape=(self.latent_dim, ))
        img = self.generator(z)
        self.discriminator.trainable = False  # Discriminator should be fixed during generator training
        validity = self.discriminator(img)
        self.combined = Model(inputs=z, outputs=validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=self._optimizer)

    def _generate_generator_input(self, batch_size):
        return np.random.normal(0, 1, (batch_size, self.latent_dim))

    @staticmethod
    def _label_discriminator_data(img_real, img_generated):
        y_real = np.ones((img_real.shape[0], 1))
        y_generated = np.zeros((img_generated.shape[0], 1))
        return y_real, y_generated

    def _get_real_imgs(self, batch_size, mode='all'):
        p = self.path + '/' + mode
        files = np.random.choice(os.listdir(p), size=batch_size, replace=True)
        X = [np.asarray(Image.open(p + '/' + file).convert('L')) / 255. for file in files]
        X = np.concatenate([x.reshape((1,) + self.img_size) for x in X])
        return X

    def _get_fake_imgs(self, batch_size):
        gen_input = self._generate_generator_input(batch_size=batch_size)
        return self.generator.predict_on_batch(gen_input)

    def _generate_discriminator_trainingset(self, batch_size):
        img_real = self._get_real_imgs(batch_size=batch_size//2)
        img_generated = self._get_fake_imgs(batch_size=batch_size//2)
        y_real, y_generated = self._label_discriminator_data(img_real, img_generated)
        x = np.concatenate([img_real, img_generated])
        y = np.concatenate([y_real, y_generated])
        p = np.random.permutation(batch_size)
        return x[p, ...], y[p, ...]

    def save_img(self, epoch, r=5, c=5):
        noise = self._generate_generator_input(batch_size=r*c)
        gen_imgs = self.generator.predict_on_batch(noise)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/imgs_epoch_%d.pdf" % epoch, format='pdf')
        plt.close()

    def train(self, epochs, batch_size=128, img_interval=50, save_interval=50, mini_epochs=50, X_train=None):
        if X_train is None:
            assert self.path is not None, "No training data available!"

        for epoch in range(epochs+1):
            # Train discriminator
            D_LOSS = []
            D_ACC = []
            for mini_epoch in range(mini_epochs):
                if X_train is None:
                    batch_x, batch_y = self._generate_discriminator_trainingset(batch_size=batch_size)
                else:
                    idx = np.random.choice(X_train.shape[0], batch_size//2)
                    batch_x = np.concatenate([X_train[idx], self._get_fake_imgs(batch_size=batch_size//2)])
                    batch_y = np.concatenate([np.ones((batch_size//2, 1)), np.zeros((batch_size//2, 1))])
                    p = np.random.permutation(batch_size)
                    batch_x = batch_x[p, ...]
                    batch_y = batch_y[p, ...]

                d_loss = self.discriminator.train_on_batch(batch_x, batch_y)
                D_LOSS.append(d_loss[0])
                D_ACC.append(d_loss[1])

            # Train generator
            G_LOSS = []
            for mini_epoch in range(mini_epochs):
                z = self._generate_generator_input(batch_size=batch_size)
                valid = np.ones((batch_size, 1))
                g_loss = self.combined.train_on_batch(z, valid)
                G_LOSS.append(g_loss)

            print("%d [D loss: %f, acc.: %.4f%%] [G loss: %f]" % (epoch, np.mean(D_LOSS), np.mean(D_ACC), np.mean(G_LOSS)), flush=True)

            if epoch % img_interval == 0:
                self.save_img(epoch)

            if epoch % save_interval == 0:
                self.generator.save("models/generator_epoch_%d.h5" % epoch)
                self.discriminator.save("models/discriminator_epoch_%d.h5" % epoch)
