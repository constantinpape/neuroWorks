# implementing different generator methods given h5 input

import h5py
import numpy as np

class DataGenerator(object):
    """
    Basic generator API.
    Implemented as python generator.
    """

    def __init__(self,
            n_batch,
            data_path,
            data_key = 'data',
            first_dim_changing = True,
            n_channels = 1,
            validation_samples = []):
        """
        Init the generator.
        @param n_batch: Number of images in batch.
        @param data_path: Path to the file containing the raw data.
        @param data_key:  Key to the data file.
        @param first_dim_changing: Boolean that determines whether the first or last dim is
        changing along instances.
        @param n_channels: Number of channels.
        @param List of samples in the training data that are used for validation.
        """
        self.n_batch    = n_batch
        self.n_channels = n_channels
        self.validation_samples = validation_samples

        self.data_path = data_path
        self.data_key  = data_key

        self.first_dim_changing = first_dim_changing

        with h5py.File(self.data_path) as f:
            if self.first_dim_changing:
                self.data_shape = f[self.data_key].shape[1:] + (self.n_channels,)
            else:
                self.data_shape = f[self.data_key].shape[:-1] + (self.n_channels,)

        self.n_instances = self._get_num_instances(self.data_path,
                self.data_key)

        self.cycle_index = 0
        # TODO check with Nasim if it makes sense to randomize in this way!
        self.permutation = np.random.permutation(self.n_instances)



    def _next_data(self):
        """
        Return the next data point.
        Overload this!
        @return: next datat point.
        """
        raise AttributeError("DataGenerator is only base class.")


    def __iter__(self):
        """
        Yield the next data point.
        """
        while True:
            yield self._next_data()


    def _get_num_instances(self, path, key):
        """
        Get number of instances in the dataset
        @param path: path to file.
        @param key:  key to file.
        @return: number of instances in file.
        """

        with h5py.File(path) as f:
            shape = f[key].shape
        if self.first_dim_changing:
            return shape[0]
        return shape[-1]


    # TODO which normalization is best? - using zero mean, 0.5 std for now
    # Do we need to normalize if we use BatchNormalizationLayers?
    # TODO When should we normalize, is it ok to normalize the batch?
    def _normalize(self, batch_data):
        """
        Normalize the batch data.
        @param batch_data: Batch data as ndarray.
        @return: Normalized batch data.
        """
        return (batch_data - np.mean(batch_data)) / np.std(batch_data)


    def _reset_cycle(self):
        """
        Reset cycling.
        """
        self.cycle_index = 0
        self.permutation = np.random.permutation(self.n_instances)


class PlainTrainDataGenerator(DataGenerator):
    """
    Generator for plain training data.
    """

    def __init__(self,
            n_batch,
            data_path,
            labels_path,
            data_key = 'data',
            labels_key = 'data',
            first_dim_changing = True):
        """
        Init the generator.
        @param n_batch: Number of images in batch.
        @param data_path: Path to the file containing the raw data.
        @param labels_path: Path to the file containing the raw labels.
        @param data_key:  Key to the data file.
        @param labels_key:  Key to the labels file.
        @param first_dim_changing: Boolean that determines whether the first or last dim is
        changing along instances.
        """

        super(PlainTrainDataGenerator, self).__init__(n_batch,
                data_path,
                data_key,
                first_dim_changing)

        self.labels_path = labels_path
        self.labels_key  = labels_key

        assert self._get_num_instances(self.labels_path, self.labels_key) == self.n_instances


    def _next_data(self):
        """
        Return next data point.
        @return: data and labels as ndarray
        """

        x = np.zeros( (self.n_batch,) + self.data_shape )
        labels_shape = (self.n_batch,) + self.data_shape[:-1] + (2,)
        y = np.zeros( labels_shape, dtype = bool )

        with h5py.File(self.data_path) as f_data,\
            h5py.File(self.labels_path) as f_labels:

            ds_data = f_data[self.data_key]
            ds_labels = f_labels[self.labels_key]

            for i in xrange(self.n_batch):

                index = self.permutation[self.cycle_index]
                # we skip the sample if they are in out validation set
                if index in self.validation_samples:
                    while index in self.validation_samples:
                        self.cycle_index += 1
                        if self.cycle_index % self.n_instances == 0:
                            self._reset_cycle()
                        index = self.permutation[self.cycle_index]

                if self.first_dim_changing:
                    x[i,...,0] = ds_data[index]
                    y[i,...,0] = ds_labels[index]
                else:
                    x[i,...,0] = ds_data[...,index]
                    y[i,...,0] = ds_labels[...,index]

                # TODO should we normalize here
                x = self._normalize(x)

                # second label channel is just the first one inverted
                y[i,...,1] = np.logical_not(y[i,...,1])

                self.cycle_index += 1
                if self.cycle_index % self.n_instances == 0:
                    self._reset_cycle()

        # change back to float TODO that is not really elegant...
        y = y.astype('float')
        return x,y


    def get_validation_samples(self):
        """
        Returns the validation samples.
        @returns: Arrays containing validation data and labels.
        @raises: RuntimeError if no validation samples were specified.
        """

        n_validation = len(self.validation_samples)
        if n_validation == 0:
            raise RuntimeError("No validation samples specified")

        x = np.zeros( (n_validation,) + self.data_shape )
        labels_shape = (n_validation,) + self.data_shape[:-1] + (2,)
        y = np.zeros( labels_shape, dtype = bool )

        with h5py.File(self.data_path) as f_data,\
            h5py.File(self.labels_path) as f_labels:

            ds_data = f_data[self.data_key]
            ds_labels = f_labels[self.labels_key]

            for i, index in enumerate(self.validation_samples):

                if self.first_dim_changing:
                    x[i,...,0] = ds_data[index]
                    y[i,...,0] = ds_labels[index]
                else:
                    x[i,...,0] = ds_data[...,index]
                    y[i,...,0] = ds_labels[...,index]

                # TODO should we normalize here
                x = self._normalize(x)

                # second label channel is just the first one inverted
                y[i,...,1] = np.logical_not(y[i,...,1])

                self.cycle_index += 1
                if self.cycle_index % self.n_instances == 0:
                    self._reset_cycle()

        # change back to float TODO that is not really elegant...
        y = y.astype('float')
        return x,y




# TODO generators for test data
# TODO generators with data augmentation
