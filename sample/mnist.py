"""
Description
-----------

Focked from Nealson's repository (Python3 version), with some modification.
"""

import pickle
import gzip
import numpy as np
from sklearn.utils import shuffle




def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('../dat/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper(one_hot_y=False):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    XXX
    """
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = (training_inputs, training_results)

    if one_hot_y:
        validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
        validation_results = [vectorized_result(y) for y in va_d[1]]
        validation_data = (validation_inputs, validation_results)

        test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
        test_results = [vectorized_result(y) for y in te_d[1]]
        test_data = (test_inputs, test_results)

    else:
        validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
        validation_data = (validation_inputs, va_d[1])
        test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
        test_data = (test_inputs, te_d[1])

    return (training_data, validation_data, test_data)



def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e



class MNIST(object):
    """ Utils of loading, processing, and batch-emitting of MNIST dataset.

    The MNIST are (28, 28)-pixal images.

    Args:
        noise_std:
            `float`, as the standard derivative of Gaussian noise that is to
            add to output `y`.
        batch_size:
            `int`, as the size of mini-batch of training data. We employ no
            mini-batch for test data.
        dtype:
            Numpy `Dtype` object of `float`, optional. As the dtype of output
            data. Default is `np.float32`.
        seed:
            `int` or `None`, optional. If `int`, then set the random-seed of
            the noise in the output data. If `None`, do nothing. This arg is
            for debugging. Default is `None`.

    Attributes:
        training_data:
            Tuple of three numpy arries, as `x`, `y`, and `y_error` for training
            data, with shape `(50000, 784)`, `(50000, 10)`, and `(50000, 10)`
            respectively.
        validation_data:
            Tuple of three numpy arries, as `x`, `y`, and `y_error` for
            validation data, with shape `(10000, 784)`, `(10000,)`, and
            `(10000,)` respectively.
        test_data:
            Tuple of three numpy arries, as `x`, `y`, and `y_error` for test
            data, with shape `(10000, 784)`, `(10000)`, and `(10000)`
            respectively.

    Methods:
        XXX
    """

    def __init__(self, noise_std, batch_size,
                 dtype=np.float32, seed=None,
                 verbose=True):

        self._dtype = dtype
        self.batch_size = batch_size

        if seed is not None:
            np.random.seed(seed)

        training_data, validation_data, test_data = load_data_wrapper()

        # Preprocess training data
        x_tr, y_tr = training_data
        x_tr = self._preprocess(x_tr)
        y_tr = self._preprocess(y_tr)
        y_err_tr = noise_std * np.ones(y_tr.shape, dtype=self._dtype)
        self.training_data = (x_tr, y_tr, y_err_tr)

        self.n_data = x_tr.shape[0]
        self.n_batches_per_epoch = round(self.n_data/self.batch_size)


        # Preprocess training data
        x_va, y_va = validation_data
        x_va = self._preprocess(x_va)
        y_va = self._preprocess(y_va)
        y_err_va = 0.0 * np.ones(y_va.shape, dtype=self._dtype)
        self.validation_data = (x_va, y_va, y_err_va)


        # Preprocess test data
        x_te, y_te = test_data
        x_te = self._preprocess(x_te)
        y_te = self._preprocess(y_te)
        y_err_te = 0.0 * np.ones(y_te.shape, dtype=dtype)
        self.test_data = (x_te, y_te, y_err_te)


    def _preprocess(self, data):
        """ Preprocessing MNIST data, including converting to numpy array,
            re-arrange the shape and dtype.

        Args:
            data:
                Any element of the tuple as the output of calling
                `mnist_loader.load_data_wrapper()`.

        Returns:
            The preprocessed, as numpy array. (This copies the input `data`,
            so that the input `data` will not be altered.)
        """
        data = np.asarray(data, dtype=self._dtype)
        data = np.squeeze(data)
        return data


    def batch_generator(self):
        """ A generator that emits mini-batch of training data, by acting
            `next()`.

        Returns:
            Tuple of three numpy arraies `(x, y, y_error)`, for the inputs of the
            model, the observed outputs of the model , and the standard derivatives
            of the observation, respectively. They are used for training only.
        """
        x, y, y_err = self.training_data
        batch_size = self.batch_size
        n_data = self.n_data

        while True:
            x, y, y_err = shuffle(x, y, y_err)  # XXX: copy ???

            for k in range(0, n_data, batch_size):
                mini_batch = (x[k:k+batch_size],
                                y[k:k+batch_size],
                                y_err[k:k+batch_size])
                yield mini_batch




if __name__ == '__main__':

    """ Test. """

    mnist_ = MNIST(noise_std=0.1, batch_size=128)
