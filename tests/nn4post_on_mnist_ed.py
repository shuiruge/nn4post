#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Shallow neural network by `Edward ` on MNIST dataset.

This employs the mini-batch and scaling, c.f. [here](http://edwardlib.org/api/\
inference-data-subsampling) and [here](http://edwardlib.org/tutorials/batch-\
training) provides an explicit instance.


Remark
------
Running in `eshell` will raise `UnicodeEncodeError`; however, run it in `bash`
instead ceases this problem.


Conclusion
----------
Implausibly costy on both CPU and RAM, even for the case:

        n_cats = 2
        n_hiddens = 10
        n_samples = 10

for which without `Mixture` all is fast and cheap (c.f. "./edward_on_mnist.py".).


TODO
----
Go back to pure TensorFlow version of nn4post (with the broadcasting trick)?
The old pure TF verison employed the Bayesian classes of TF itself. I can write
a class that inherit `edward.VariationalInference`, but with the TF's `update()`
method.
"""


import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import (
    Normal, Categorical, Mixture,
    NormalWithSoftplusScale)
import sys
sys.path.append('../sample/')
from sklearn.utils import shuffle
import mnist_loader



ed.set_seed(42)  # for debugging.



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
        x_train:
            Numpy array with shape `(10000, 784, 1)` and dtype `dtype`.
        y_train:
            Numpy array with shape `(10000, 10, 1)` and dtype `dtype`.
        x_test:
            Numpy array with shape `(10000, 784, 1)` and dtype `dtype`.
        y_test:
            Numpy array with shape `(10000,)` and dtype `dtype`.
        XXX


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

        training_data, validation_data, test_data = \
            mnist_loader.load_data_wrapper()

        # Preprocess training data
        x_tr, y_tr = training_data
        x_tr = self._preprocess(x_tr)
        y_tr = self._preprocess(y_tr)
        y_err_tr = noise_std * np.ones(y_tr.shape, dtype=self._dtype)
        self.training_data = (x_tr, y_tr, y_err_tr)

        self.n_data = len(x_tr)

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



noise_std = 0.1
batch_size = 16  # test!
mnist = MNIST(noise_std, batch_size)



# MODEL
with tf.name_scope("model"):


    # -- This sets the priors of model parameters. I.e. the :math:`p(\theta)`
    #    in the documentation.
    #
    #    Notice that the priors of the biases shall be uniform on
    #    :math:`\mathbb{R}`; herein we use `Normal()` with large `scale`
    #    (e.g. `100`) to approximate it.
    n_inputs = 28 * 28  # number of input features.
    n_hiddens = 10  # number of perceptrons in the (single) hidden layer.
    n_outputs = 10  # number of perceptrons in the output layer.
    w_h = Normal(loc=tf.zeros([n_inputs, n_hiddens]),
                 scale=tf.ones([n_inputs, n_hiddens]),
                 name="w_h")
    w_a = Normal(loc=tf.zeros([n_hiddens, n_outputs]),
                 scale=tf.ones([n_hiddens, n_outputs]),
                 name="w_a")
    b_h = Normal(loc=tf.zeros([n_hiddens]),
                 scale=tf.ones([n_hiddens]) * 100,
                 name="b_h")
    b_a = Normal(loc=tf.zeros([n_outputs]),
                 scale=tf.ones([n_outputs]) * 100,
                 name="b_a")


    # -- Placeholder for input data.
    x = tf.placeholder(tf.float32, [None, n_inputs],
                       name='x')
    y_error = tf.placeholder(tf.float32, [None, n_outputs],
                             name='y_error')


    # -- The model architecture. This is for getting the likelihood.
    def neural_network(x):
        """
        Args:
            x:
                Tensor object with shape `[None, n_inputs]` and dtype
                `tf.float32`.

        Returns:
            Tensor object with shape `[None, n_outputs]` and dtype `tf.float32`.
        """
        # shape: `[None, n_hiddens]`
        hidden = tf.sigmoid(tf.matmul(x, w_h) + b_h)
        # shape: `[None, n_outputs]`
        activation = tf.nn.softmax(tf.matmul(hidden, w_a) + b_a)
        return activation
    prediction = neural_network(x)

    # -- This sets the likelihood. I.e. the :math:`p( D \mid \theta )`
    #    in the documentation.
    #    Precisely, this gives :math:`p( y \mid w, b; x)`, where :math:`(w, b)`
    #    is the :math:`\theta`, and :math:`x` is deterministic and fixed
    #    throughout the process of Bayesian inference.
    #    (The `0.1` may not be equal to the `noise_std` in the data, which we
    #     do not known. This number of a prior in fact.)
    y = Normal(loc=prediction,  # recall shape: `[n_data, 1]`.
               scale=y_error,
               name="y")



# INFERENCE
with tf.name_scope("posterior"):


    # -- This sets the :math:`q( \theta \mid \nu )`, where :math:`\nu` is as
    #    the :math:`(a, \mu, \zeta)` in the documentation. The `loc`s and
    #    `scale`s are variables that is to be tuned in every iteration of
    #    optimization (by `Inference.update()`).
    #
    #    CAUTION:
    #        using `ed.models.MultivariateNormalDiag` (the same as the
    #        `tf.contrib.distributions.MultivariateNormalDiag`) makes the
    #        inference extremely slow, comparing with using `ed.models.Normal`.
    #        The reason is unclear.

    # -- We set all compnents of parameter-space have independent distributions.
    #    This is essential for generalizing (transfering) the trained model
    #    (posterior, data-encoding, or whatever you'd like to call).
    #
    #    The trick here is that `Categorical` supports broadcasting. I.e., the
    #    `[:-1]` dimensions of the argument of `Categorical()` are for the
    #    broadcasting, and the last dimension for categorical classes.
    n_cats = 1
    var = {
        'cat': {},  # type: `Tensor`.
        'locs': {},  # type: list of `Tensor`s.
        'scales': {},  # type: list of `Tensor`s.
        }

    with tf.name_scope("qw_h"):
        var['cat']['qw_h'] = tf.Variable(
            tf.zeros([n_inputs, n_hiddens, n_cats]),
            name='cat')
        var['locs']['qw_h'] = [
            tf.Variable(
                tf.random_normal([n_inputs, n_hiddens]),
                name='loc_{0}'.format(i))
            for i in range(n_cats)]
        var['scales']['qw_h'] = [
            tf.Variable(
                tf.random_normal([n_inputs, n_hiddens]),
                name='scale_{0}'.format(i))
            for i in range(n_cats)]
        qw_h = Mixture(
            cat=Categorical(logits=var['cat']['qw_h']),
            components=[
                NormalWithSoftplusScale(
                    loc=var['locs']['qw_h'][i],
                    scale=var['scales']['qw_h'][i])
                for i in range(n_cats)])

    with tf.name_scope("qw_a"):
        var['cat']['qw_a'] = tf.Variable(
            tf.zeros([n_hiddens, n_outputs, n_cats]),
            name='cat')
        var['locs']['qw_a'] = [
            tf.Variable(
                tf.random_normal([n_hiddens, n_outputs]),
                name='loc_{0}'.format(i))
            for i in range(n_cats)]
        var['scales']['qw_a'] = [
            tf.Variable(
                tf.random_normal([n_hiddens, n_outputs]),
                name='scale_{0}'.format(i))
            for i in range(n_cats)]
        qw_a = Mixture(
            cat=Categorical(logits=var['cat']['qw_a']),
            components=[
                NormalWithSoftplusScale(
                    loc=var['locs']['qw_a'][i],
                    scale=var['scales']['qw_a'][i])
                for i in range(n_cats)])

    with tf.name_scope("qb_h"):
        var['cat']['qb_h'] = tf.Variable(
            tf.zeros([n_hiddens, n_cats]),
            name='cat')
        var['locs']['qb_h'] = [
            tf.Variable(
                tf.random_normal([n_hiddens]),
                name='loc_{0}'.format(i))
            for i in range(n_cats)]
        var['scales']['qb_h'] = [
            tf.Variable(
                tf.random_normal([n_hiddens]),
                name='scale_{0}'.format(i))
            for i in range(n_cats)]
        qb_h = Mixture(
            cat=Categorical(logits=var['cat']['qb_h']),
            components=[
                NormalWithSoftplusScale(
                    loc=var['locs']['qb_h'][i],
                    scale=var['scales']['qb_h'][i])
                for i in range(n_cats)])

    with tf.name_scope("qb_a"):
        var['cat']['qb_a'] = tf.Variable(
            tf.zeros([n_outputs, n_cats]),
            name='cat')
        var['locs']['qb_a'] = [
            tf.Variable(
                tf.random_normal([n_outputs]),
                name='loc_{0}'.format(i))
            for i in range(n_cats)]
        var['scales']['qb_a'] = [
            tf.Variable(
                tf.random_normal([n_outputs]),
                name='scale_{0}'.format(i))
            for i in range(n_cats)]
        qb_a = Mixture(
            cat=Categorical(logits=var['cat']['qb_a']),
            components=[
                NormalWithSoftplusScale(
                    loc=var['locs']['qb_a'][i],
                    scale=var['scales']['qb_a'][i])
                for i in range(n_cats)])



# PLAY
# Set the parameters of training
logdir = '../dat/logs'
n_batchs = mnist.batch_size
n_epochs = 1
n_samples = 10
scale = {y: mnist.n_data / mnist.batch_size}
y_ph = tf.placeholder(tf.float32, [None, n_outputs],
                      name='y')
inference = ed.KLqp(latent_vars={w_h: qw_h, b_h: qb_h,
                                 w_a: qw_a, b_a: qb_a},
                    data={y: y_ph})
inference.initialize(
    n_iter=n_batchs * n_epochs,
    n_samples=n_samples,
    scale=scale,
    logdir=logdir)
tf.global_variables_initializer().run()


sess = ed.get_session()
profiling = True


if profiling:
    # -- C.f. `help(tf.profiler.Profiler)`.
    profiler = tf.profiler.Profiler(sess.graph)

    for i in range(inference.n_iter):

        batch_generator = mnist.batch_generator()
        x_batch, y_batch, y_error_batch = next(batch_generator)
        feed_dict = {x: x_batch,
                    y_ph: y_batch,
                    y_error: y_error_batch}


        # With profiler
        run_meta = tf.RunMetadata()
        _, t, loss, summary = sess.run(
            [ inference.train, inference.increment_t,
              inference.loss, inference.summarize ],
            feed_dict,
            options=tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE),
            run_metadata=run_meta)
        profiler.add_step(i, run_meta)

        # Profile the parameters of your model.
        option_builder = tf.profiler
        profiler.profile_name_scope(options=(option_builder.ProfileOptionBuilder
                                            .trainable_variables_parameter()))

        # Or profile the timing of your model operations.
        opts = option_builder.ProfileOptionBuilder.time_and_memory()
        profiler.profile_operations(options=opts)

        # Or you can generate a timeline:
        opts = (option_builder.ProfileOptionBuilder(
                option_builder.ProfileOptionBuilder.time_and_memory())
                    .with_step(i)
                    .with_timeline_output('timeline').build())
        profiler.profile_graph(options=opts)


        info_dict = {'t': t, 'loss': loss}
        inference.print_progress(info_dict)

    # Auto detect problems and generate advice.
    #profiler.advise()


else:
    for i in range(inference.n_iter):

        batch_generator = mnist.batch_generator()
        x_batch, y_batch, y_error_batch = next(batch_generator)
        feed_dict = {x: x_batch,
                    y_ph: y_batch,
                    y_error: y_error_batch}


        _, t, loss = sess.run(
            [ inference.train, inference.increment_t,
              inference.loss ],
            feed_dict)

        info_dict = {'t': t, 'loss': loss}
        inference.print_progress(info_dict)





'''
# EVALUATE
# -- That is, check your result.
x_test, y_test, y_error_test = mnist.test_data
n_test_data = len(y_test)
print('{0} test data.'.format(n_test_data))

prediction_post = ed.copy(prediction, {w_h: qw_h, b_h: qb_h,
                                       w_a: qw_a, b_a: qb_a})
n_models = 500  # number of Monte Carlo neural network models.
# shape: [n_models, n_test_data, n_outputs]
prediction_vals = [prediction_post.eval(feed_dict={x: x_test})
                   for i in range(n_models)]

acc = 0
for i, y_true in enumerate(y_test):
    # shape: [n_samples, n_outputs]
    softmax_vals = np.array([pred[i] for pred in prediction_vals])
    # shape: [n_outputs]
    mean_softmax_val = np.mean(softmax_vals, axis=0)
    pred_val = np.argmax(mean_softmax_val)
    if int(pred_val) == int(y_true):
        acc += 1

print('Accuracy on test data: {0} %'.format(acc / n_test_data * 100))
'''
