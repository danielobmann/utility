import tensorflow as tf
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import *
from PIL import Image
import os


class CustomObjects:
    def __init__(self, sess=None):
        if sess is None:
            self.sess = K.get_session()
        else:
            self.sess = sess
        self.custom_objects = {'bound': self.bound, 'KerasPSNR': self.KerasPSNR, 'KerasNMSE': self.KerasNMSE,
                               'loss': self.wavelet_loss(alpha=0.5), 'l2_loss': self.l2_loss}
        pass

    @staticmethod
    def project(x):
        return np.clip(x, 0, 1)

    @staticmethod
    def bound(x):
        return K.minimum(x, K.ones_like(x))

    @staticmethod
    def KerasPSNR(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)

    @staticmethod
    def KerasNMSE(y_true, y_pred):
        m = K.mean(K.square(y_true - y_pred))
        n = K.mean(K.square(y_true))
        return m/n

    @staticmethod
    def KerasSSIM(y_true, y_pred):
        y_true_bar, y_pred_bar = K.mean(y_true), K.mean(y_pred)
        sigma_true, sigma_pred = K.var(y_true), K.var(y_pred)
        cov = K.mean((y_true - y_true_bar)*(y_pred - y_pred_bar))
        c1, c2 = 0.01 * K.max(y_true), 0.03 * K.max(y_true)
        return (2*y_true_bar*y_pred_bar + c1)*(2*cov + c2)/((y_true_bar**2 + y_pred_bar**2 + c1)*(sigma_true + sigma_pred + c2))

    @staticmethod
    def PSNR(x, xhat):
        maxvalue = np.amax(x)
        return 10 * np.log10(maxvalue**2 / np.mean((x - xhat) ** 2))

    @staticmethod
    def NMSE(x, x_hat):
        error = np.mean((x - x_hat)**2)
        normalizer = np.mean(x**2)
        return error/normalizer

    @staticmethod
    def SSIM(x, xhat):
        xbar, xhatbar = np.mean(x), np.mean(xhat)
        sigma, sigmahat = np.var(x), np.var(xhat)
        cov = np.sum((x - xbar)*(xhat - xhatbar))/(np.prod(x.shape) - 1)
        c1, c2 = 0.01*np.max(x), 0.03*np.max(x)
        return (2*xbar*xhatbar + c1)*(2*cov + c2)/((xbar**2 + xhatbar**2 + c1)*(sigma + sigmahat + c2))

    @staticmethod
    def check_dim(x0):
        if len(np.asarray(x0).shape) != 4:
            return np.asarray(x0)[None, ..., None]
        else:
            return x0
        pass

    @staticmethod
    def plot(x, text=[], colorbar=True, axis=False, cmap='gray', col='orange', save=None, show=False, title=None):
        fig, ax = plt.subplots()
        im = ax.imshow(x, cmap=cmap, vmin=0.0, vmax=1.0)
        
        if colorbar:
            fig.colorbar(im)
            
        if not axis:
            ax.axis('off')
            
        xstart = 0.01
        ystart = 1-0.05
        ystep = 0.05
        
        for l in range(len(text)):
            t = text[l]
            ax.text(xstart, ystart - l*ystep, t, transform=ax.transAxes, color=col)
        
        if not (save is None):
            fig.savefig(save, format='pdf')
            
        if show:
            fig.show()
            
        elif not show:
            fig.clf()
            
        if not(title is None):
            ax.set_title(title)
        
        pass

    @staticmethod
    def multiplot(X, axis=False, cmap='gray', show=True, **kwargs):
        assert len(X) % 2 == 0, "Should have an even number of images"

        ncol = len(X) // 2

        fig, ax = plt.subplots(ncol, 2)
        for i in range(len(X)):
            ax[i].imshow(X[i], cmap=cmap, **kwargs)
            if not axis:
                ax[i].axis('off')

        if show:
            fig.show()
        pass

    @staticmethod
    def _mark_inset(parent_axes, inset_axes, **kwargs):
        # This code is copied from the matplotlib source code and slightly modified.
        # This is done to avoid the 'connection lines'.
        rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

        if 'fill' in kwargs:
            pp = BboxPatch(rect, **kwargs)
        else:
            fill = bool({'fc', 'facecolor', 'color'}.intersection(kwargs))
            pp = BboxPatch(rect, fill=fill, **kwargs)
        parent_axes.add_patch(pp)

        p1 = BboxConnector(inset_axes.bbox, rect, loc1=1, **kwargs)
        p1.set_clip_on(False)

        p2 = BboxConnector(inset_axes.bbox, rect, loc1=1, **kwargs)
        p2.set_clip_on(False)

        return pp, p1, p2

    def zoomed_plot(self, x, xlim, ylim, zoom=2, text=None, textloc=[], fsize=18, cmap='gray'):

        color = 'orange'
        fig, ax = plt.subplots()
        ax.imshow(np.flipud(x), cmap=cmap, vmin=0.0, vmax=1.0, origin="lower")
        ax.axis('off')

        axins = zoomed_inset_axes(ax, zoom, loc=4)

        axins.set_xlim(xlim[0], xlim[1])
        axins.set_ylim(ylim[0], ylim[1])

        self._mark_inset(ax, axins, fc='none', ec=color)

        axins.imshow(np.flipud(x), cmap=cmap, vmin=0.0, vmax=1.0, origin="lower")
        axins.patch.set_edgecolor(color)
        axins.patch.set_linewidth('3')
        axins.set_xticks([], [])
        axins.set_yticks([], [])
        #axins.axis('off')

        if not (text is None):
            ax.text(textloc[0], textloc[1], text, color=color, fontdict={'size': fsize}, transform=ax.transAxes)
        pass

    @staticmethod
    def error_plot(err, name, path):
        plt.semilogy(err)
        plt.title(name)
        plt.savefig(path + name + '.pdf', format='pdf')
        plt.clf()
        pass

    @staticmethod
    def wavelet_loss(alpha=0.5):
        H = (1 / np.sqrt(2)) * np.array([-1, 1], dtype=np.float32)
        L = (1 / np.sqrt(2)) * np.array([1, 1], dtype=np.float32)

        l = np.outer(L, L).reshape((2, 2, 1, 1))
        d = np.outer(H, H).reshape((2, 2, 1, 1))
        v = np.outer(L, H).reshape((2, 2, 1, 1))
        h = np.outer(H, L).reshape((2, 2, 1, 1))

        def loss(y_true, y_pred):
            diff = (y_true - y_pred)
            low = K.conv2d(diff, l, strides=(2, 2))
            diag = K.conv2d(diff, d, strides=(2, 2))
            vert = K.conv2d(diff, v, strides=(2, 2))
            hor = K.conv2d(diff, h, strides=(2, 2))
            return K.sum(K.mean(alpha * K.square(low) + K.square(diag) + K.square(vert) + K.square(hor), axis=0))

        return loss

    @staticmethod
    def l2_loss(y_true, y_pred):
        return K.sum(K.mean(K.square(y_true - y_pred), axis=0))

    @staticmethod
    def get_random_image(path, mode='train', name=False):
        p = path + '/' + mode
        img = np.random.choice(os.listdir(p), size=1)[0]
        if name:
            print(img)
        img = np.asarray(Image.open(p + '/' + img).convert('L'))/255.
        return img

