import odl
import numpy as np
import scipy


class ForwardOperators:

    def __init__(self, img_height=512, img_width=512, img_channels=1):
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels

    def radon(self, n_theta, n_s, limited_angle=0, filter_type='Hann', **kwargs):
        reco_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1],
                                       shape=[self.img_height, self.img_width], dtype='float32')
        angle_partition = odl.uniform_partition(limited_angle,
                                                (np.pi-limited_angle) * (1 - 1/(2*n_theta)),
                                                n_theta)
        detector_partition = odl.uniform_partition(-1.5, 1.5, n_s)
        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

        radon = odl.tomo.RayTransform(reco_space, geometry)
        fbp = odl.tomo.fbp_op(radon, filter_type=filter_type, **kwargs)

        return radon, fbp

    def get_pseudoinverse(self, operator, k=100, tol=0):
        """
        Compute pseudoinverse of linear odl operator using SVD with k singular values.
        """

        a = odl.operator.oputils.as_scipy_operator(operator)
        u, s, v = scipy.sparse.linalg.svds(a, k=k, tol=tol)

        assert (s != 0).all(), "Zero singular values computed. Choose smaller k value!"

        def pseudoinverse(y):
            ret = np.dot(u.transpose(), y.flatten())/s
            return np.dot(v.transpose(), ret).reshape((self.img_height, self.img_width))

        return pseudoinverse
