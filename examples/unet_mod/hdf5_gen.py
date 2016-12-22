import numpy as np
import h5py
import os

from image_util import BaseDataProvider


class Hdf5DataProvider2D(BaseDataProvider):
    """
    Provider for 2D gray scale images stacked in a hdf5 array.
    Assumes (x,y,images).
    :param raw_path: path to the raw images
    :param labels_path: (optional) path to the labels
    """

    channels = 1
    n_class = 2

    def __init__(self, raw_path, labels_path = None, raw_key = 'data', labels_key = 'data'):
        super(Hdf5DataProvider2D, self).__init__()

        assert os.path.exists(raw_path), raw_path
        self.raw_path = raw_path
        self.raw_key  = raw_key

        with h5py.File(raw_path) as f:
            assert self.raw_key in f.keys(), str(f.keys())
            shape = f[self.raw_key].shape
            assert len(shape) == 3, str(len(shape))
            self.shape = shape[:2]
            self.n_images = shape[2]

        if labels_path is not None:
            assert os.path.exists(labels_path), labels_path
            with h5py.File(labels_path) as f:
                assert labels_key in f.keys(), str(f.keys())
                shape  = f[labels_key].shape
                assert len(shape) == 3, str(len(shape))
                assert shape[:2] == self.shape
                assert shape[2] == self.n_images

        self.labels_path = labels_path
        self.labels_key  = labels_key

        self.img_idx = 0


    def _next_data(self):
        self.img_idx = self.img_idx % self.n_images

        image = np.zeros(self.shape)
        label = np.zeros(self.shape, np.bool)

        with h5py.File(self.raw_path) as f:
            image = f[self.raw_key][:,:,self.img_idx]

        if self.labels_path is not None:
            with h5py.File(self.labels_path) as f:
                labels = f[self.labels_key][:,:,self.img_idx]

        self.img_idx += 1
        return image, label
