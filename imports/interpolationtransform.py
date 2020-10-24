import odl
import numpy as np
import scipy


class Transform:
    def __init__(self, n_theta=720, factor=24, img_height=512, img_width=512):
        self._n_theta = n_theta
        self._factor = factor
        self._img_height = img_height
        self._img_width = img_width
        self._n_s = int(img_height * 1.5)

        # Angles
        self._angle_partition = odl.uniform_partition(0, np.pi * (1 - 1 / (2 * n_theta)), n_theta)

        # Operators
        self.radon, self.fbp = self._construct_operators()
        self.radon_pseudoinverse = self._get_pseudoinverse()

    def _construct_operators(self):
        reco_space = odl.uniform_discr(min_pt=[-1, -1],
                                       max_pt=[1, 1],
                                       shape=[self._img_height, self._img_width],
                                       dtype='float32')

        detector_partition = odl.uniform_partition(-1.5, 1.5, self._n_s)
        geometry = odl.tomo.Parallel2dGeometry(self._angle_partition, detector_partition)
        radon = odl.tomo.RayTransform(reco_space, geometry)
        fbp = odl.tomo.fbp_op(radon, filter_type='Hann')

        return radon, fbp

    def get_data(self, x):
        data = np.asarray(self.radon(x))
        return self.subsample(data, factor=self._factor), data

    def get_single_y_input(self, x, method="zero"):
        data_sparse, data = self.get_data(x)
        data_upsample = self.upsample(data_sparse, factor=self._factor, method=method)
        return data_upsample, data

    def get_batch_y_input(self, X, method="zero"):
        inp = []
        out = []
        for x in X:
            inp_data, out_data = self.get_single_y_input(x, method=method)
            inp.append(inp_data.reshape((1, self._n_theta, self._n_s, 1)))
            out.append(out_data.reshape((1, self._n_theta, self._n_s, 1)))
        return np.concatenate(inp), np.concatenate(out)

    def _get_pseudoinverse(self, k=100):
        a = odl.operator.oputils.as_scipy_operator(self.radon)
        u, s, v = scipy.sparse.linalg.svds(a, k=k)

        def pseudoinverse(y):
            ret = np.dot(u.transpose(), np.asarray(y).flatten())/s
            ret = np.dot(v.transpose(), ret)
            return ret.reshape((512, 512))

        return pseudoinverse

    def project_image(self, y):
        x = self.radon_pseudoinverse(y)
        return np.asarray(self.radon(x))

    def project_nullspace_orthogonal(self, x):
        y = np.asarray(self.radon(x))
        return self.radon_pseudoinverse(y)

    @staticmethod
    def subsample(y, factor=4):
        return np.asarray(y)[::factor, :]

    def upsample(self, y, factor=4, method="zero"):
        s = y.shape
        ret = np.zeros((s[0] * factor, s[1]))
        if method is "zero":
            ret[::factor, :] = y
        elif method is "image":
            ret[::factor, :] = y
            self.project_image(ret)
        elif method is "linear":
            x = self._angle_partition.coord_vectors[0]
            xs = x[::factor]
            for i in range(s[1]):
                ret[:, i] = np.interp(x, xs, y[:, i])
        return ret

    @staticmethod
    def set_equal(y_low, y_high):
        ret = y_high.copy()
        factor = int(y_high.shape[0] / y_low.shape[0])
        ret[::factor, :] = y_low
        return ret

"""
# Testing corner
import matplotlib.pyplot as plt
tr = Transform()
x = np.random.uniform(0, 1, (512, 512))
y = tr.radon(x)
y_low = tr.subsample(y, factor=4)
y_up = tr.upsample(y_low, factor=4, method="linear")


np.sum((y_up - y)**2)
i = 100
plt.plot(y_up[:, i])
plt.plot(y_low[:, i])
plt.show()
"""