import os
import numpy as np


def cosine_decay(epoch, total, initial=1e-3):
    return initial/2.*(1 + np.cos(np.pi*epoch/total))


def setup_path(path, verbose=0):
    if not os.path.exists(path):
        os.mkdir(path)

        if verbose:
            print("Created new path %s." % path)
