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
"""


import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import Categorical, NormalWithSoftplusScale, Mixture
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../sample/')
from sklearn.utils import shuffle
from tools import Timer, get_accuracy, get_variable_value_dict, TimeLiner
import mnist
import time
import pickle
from KLqp_tf import KLqp
from tensorflow.python.client import timeline
import json



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # turn off the TF noise.
ed.set_seed(42)  # for debugging.



# DATA
noise_std = 0.1
batch_size = 16
mnist_ = mnist.MNIST(noise_std, batch_size)
batch_generator = mnist_.batch_generator()



# MODEL
path_to_pretrained = '../dat/pretrained/{0}.pkl'\
                     .format(__file__)

with tf.name_scope("model"):


    # -- This sets the priors of model parameters. I.e. the :math:`p(\theta)`
    #    in the documentation.
    #
    #    There are two ways of setting prior:
    #
    #        1. hand-wave;
    #
    #        2. use pre-trained posterior.
    #
    #    The later is prefered herein.

    try:

        # -- Use pre-trained posterior as prior

        print('Try to use pre-trained posterior as prior.')

        pretrained = pickle.load(open(path_to_pretrained, 'rb'))
        n_inputs, n_hiddens = pretrained['posterior/qw_h/loc:0'].shape
        n_hiddens, n_outputs = pretrained['posterior/qw_a/loc:0'].shape

        w_h = NormalWithSoftplusScale(
            loc=pretrained['posterior/qw_h/loc:0'],
            scale=pretrained['posterior/qw_h/scale:0'],
            name="w_h")
        w_a = NormalWithSoftplusScale(
            loc=pretrained['posterior/qw_a/loc:0'],
            scale=pretrained['posterior/qw_a/scale:0'],
            name="w_a")
        b_h = NormalWithSoftplusScale(
            loc=pretrained['posterior/qb_h/loc:0'],
            scale=pretrained['posterior/qb_h/scale:0'],
            name="b_h")
        b_a = NormalWithSoftplusScale(
            loc=pretrained['posterior/qb_a/loc:0'],
            scale=pretrained['posterior/qb_a/scale:0'],
            name="b_a")

    except Exception as e:

        # -- Use hand-waved priors
        #
        #    Notice that the priors of the biases shall be uniform on
        #    :math:`\mathbb{R}`; herein we use `Normal()` with large `scale`
        #    (e.g. `100`) to approximate it.

        print('WARNING - cannot use pre-trained posterior as prior:\n\t', e)
        print('Instead, use hand-waved prior.')

        n_inputs = 28 * 28  # number of input features.
        n_hiddens = 15  # number of perceptrons in the (single) hidden layer.
        n_outputs = 10  # number of perceptrons in the output layer.

        w_h = NormalWithSoftplusScale(
            loc=tf.zeros([n_inputs, n_hiddens]),
            scale=tf.ones([n_inputs, n_hiddens]),
            name="w_h")
        w_a = NormalWithSoftplusScale(
            loc=tf.zeros([n_hiddens, n_outputs]),
            scale=tf.ones([n_hiddens, n_outputs]),
            name="w_a")
        b_h = NormalWithSoftplusScale(
            loc=tf.zeros([n_hiddens]),
            scale=tf.ones([n_hiddens]) * 100,
            name="b_h")
        b_a = NormalWithSoftplusScale(
            loc=tf.zeros([n_outputs]),
            scale=tf.ones([n_outputs]) * 100,
            name="b_a")


    # -- Placeholder for input data.
    x = tf.placeholder(tf.float32, [None, n_inputs],
                       name='x')
    y_error = tf.placeholder(tf.float32, [None, n_outputs],
                             name='y_error')


    # -- The model architecture. This is for getting the likelihood.
    def neural_network(x, w_h, b_h, w_a, b_a):
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
    def model(w_h, b_h, w_a, b_a):
        return neural_network(x, w_h, b_h, w_a, b_a)
    prediction = model(w_h, b_h, w_a, b_a)

    # -- This sets the likelihood. I.e. the :math:`p( D \mid \theta )`
    #    in the documentation.
    #    Precisely, this gives :math:`p( y \mid w, b; x)`, where :math:`(w, b)`
    #    is the :math:`\theta`, and :math:`x` is deterministic and fixed
    #    throughout the process of Bayesian inference.
    #    (The `0.1` may not be equal to the `noise_std` in the data, which we
    #     do not known. This number of a prior in fact.)
    y = NormalWithSoftplusScale(
        loc=prediction,  # recall shape: `[n_data, 1]`.
        scale=y_error,
        name="y")


def without_cat():

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
        with tf.name_scope("qw_h"):
            loc_qw_h = tf.Variable(
                tf.random_normal([n_inputs, n_hiddens]),
                name='loc')
            scale_qw_h = tf.Variable(
                tf.random_normal([n_inputs, n_hiddens]),
                name='scale')
            qw_h = NormalWithSoftplusScale(loc=loc_qw_h, scale=scale_qw_h)
        with tf.name_scope("qw_a"):
            loc_qw_a = tf.Variable(
                tf.random_normal([n_hiddens, n_outputs]),
                name='loc')
            scale_qw_a = tf.Variable(
                tf.random_normal([n_hiddens, n_outputs]),
                name="scale")
            qw_a = NormalWithSoftplusScale(loc=loc_qw_a, scale=scale_qw_a)
        with tf.name_scope("qb_h"):
            loc_qb_h = tf.Variable(
                tf.random_normal([n_hiddens]),
                name="loc")
            scale_qb_h = tf.Variable(
                tf.random_normal([n_hiddens]),
                name="scale")
            qb_h = NormalWithSoftplusScale(loc=loc_qb_h, scale=scale_qb_h)
        with tf.name_scope("qb_a"):
            loc_qb_a = tf.Variable(
                tf.random_normal([n_outputs]),
                name="loc")
            scale_qb_a = tf.Variable(
                tf.random_normal([n_outputs]),
                name="scale")
            qb_a = NormalWithSoftplusScale(loc=loc_qb_a, scale=scale_qb_a)




    # PLAY
    # Set the parameters of training
    n_epochs = 1
    #n_iter = mnist_.n_batches_per_epoch * n_epochs
    n_iter = 2
    n_samples = 5
    scale = {y: mnist_.n_data / mnist_.batch_size}
    logdir = '../dat/logs'

    y_ph = tf.placeholder(tf.float32, [None, n_outputs],
                        name='y')
    inference = KLqp(model=model,
                     latent_vars={w_h: qw_h, b_h: qb_h,
                                  w_a: qw_a, b_a: qb_a},
                     data={y: y_ph})
    with Timer():
        inference.initialize(
            n_iter=n_iter,
            n_samples=n_samples,
            scale=scale,
            logdir=logdir,
            log_vars=[])
    tf.global_variables_initializer().run()


    sess = ed.get_session()


    # Add node of posterior to graph
    prediction_post = ed.copy(prediction,
                            { w_h: qw_h, b_h: qb_h,
                                w_a: qw_a, b_a: qb_a })

    #with tf.contrib.tfprof.ProfileContext('../dat/train_dir') as pctx:
    time_start = time.time()
    time_start_epoch = time_start

    for i in range(inference.n_iter):

        x_batch, y_batch, y_error_batch = next(batch_generator)
        feed_dict = {x: x_batch,
                    y_ph: y_batch,
                    y_error: y_error_batch}

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        many_runs_timeline = TimeLiner()
        with Timer():
            _, t, loss = sess.run(
                [ inference.train, inference.increment_t,
                inference.loss ],
                feed_dict,
                options=run_options,
                run_metadata=run_metadata)
        inference.train_writer.add_run_metadata(run_metadata, 'step%d' % i)

        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        many_runs_timeline.update_timeline(chrome_trace)

        # Validation for each epoch
        if (i+1) % mnist_.n_batches_per_epoch == 0:

            epoch = int( (i+1) / mnist_.n_batches_per_epoch )
            print('\nFinished the {0}-th epoch'.format(epoch))
            print('Elapsed time {0} sec.'.format(time.time()-time_start_epoch))

            # Get validation data
            x_valid, y_valid, y_error_valid = mnist_.validation_data
            x_valid, y_valid, y_error_valid = \
                shuffle(x_valid, y_valid, y_error_valid)
            x_valid, y_valid, y_error_valid = \
                x_valid[:128], y_valid[:128], y_error_valid[:128]

            # Get accuracy
            n_models = 100  # number of Monte Carlo neural network models.
            # shape: [n_models, n_test_data, n_outputs]
            softmax_vals = [prediction_post.eval(feed_dict={x: x_valid})
                            for i in range(n_models)]
            # shape: [n_test_data, n_outputs]
            mean_softmax_vals = np.mean(softmax_vals, axis=0)
            # shape: [n_test_data]
            y_pred = np.argmax(mean_softmax_vals, axis=-1)
            accuracy = get_accuracy(y_pred, y_valid)

            print('Accuracy on validation data: {0} %'\
                    .format(accuracy/mnist_.batch_size*100))
            time_start_epoch = time.time()  # re-initialize.

    many_runs_timeline.save('../dat/timelines/timeline_with_0_cats.json')
    time_end = time.time()
    print('------ Elapsed {0} sec in training.\n\n'\
          .format(time_end-time_start))



    '''
    # EVALUATE
    x_test, y_test, y_error_test = mnist_.test_data
    n_test_data = len(y_test)
    print('{0} test data.'.format(n_test_data))

    n_models = 500  # number of Monte Carlo neural network models.
    # shape: [n_models, n_test_data, n_outputs]
    prediction_vals = [prediction_post.eval(feed_dict={x: x_test})
                        for i in range(n_models)]

    # Get accuracy
    n_models = 100  # number of Monte Carlo neural network models.
    # shape: [n_models, n_test_data, n_outputs]
    softmax_vals = [prediction_post.eval(feed_dict={x: x_test})
                    for i in range(n_models)]
    # shape: [n_test_data, n_outputs]
    mean_softmax_vals = np.mean(softmax_vals, axis=0)
    # shape: [n_test_data]
    y_pred = np.argmax(mean_softmax_vals, axis=-1)
    accuracy = get_accuracy(y_pred, y_test)

    print('Accuracy on test data: {0} %'\
            .format(accuracy/mnist_.batch_size*100))
    '''

    ## Save the training result
    #pretrained = get_variable_value_dict(sess)
    #print(pretrained.keys())
    #try:
    #    pickle.dump(pretrained, open(path_to_pretrained, 'wb'))
    #except Exception as e:
    #    print('Fail in saving trained variable to disk - ', e)



def with_cat(n_cats):


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
                    name='loc/{0}'.format(i))
                for i in range(n_cats)]
            var['scales']['qw_h'] = [
                tf.Variable(
                    tf.random_normal([n_inputs, n_hiddens]),
                    name='scale/{0}'.format(i))
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
                    name='loc/{0}'.format(i))
                for i in range(n_cats)]
            var['scales']['qw_a'] = [
                tf.Variable(
                    tf.random_normal([n_hiddens, n_outputs]),
                    name='scale/{0}'.format(i))
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
                    name='loc/{0}'.format(i))
                for i in range(n_cats)]
            var['scales']['qb_h'] = [
                tf.Variable(
                    tf.random_normal([n_hiddens]),
                    name='scale/{0}'.format(i))
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
                    name='loc/{0}'.format(i))
                for i in range(n_cats)]
            var['scales']['qb_a'] = [
                tf.Variable(
                    tf.random_normal([n_outputs]),
                    name='scale/{0}'.format(i))
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
    n_epochs = 1
    #n_iter = mnist_.n_batches_per_epoch * n_epochs
    n_iter = 2  # test!
    n_samples = 5
    scale = {y: mnist_.n_data / mnist_.batch_size}
    logdir = '../dat/logs'

    y_ph = tf.placeholder(tf.float32, [None, n_outputs],
                          name='y')
    inference = KLqp(model=model,
                     latent_vars={w_h: qw_h, b_h: qb_h,
                                  w_a: qw_a, b_a: qb_a},
                     data={y: y_ph})
    with Timer():
        inference.initialize(
            n_iter=n_iter,
            n_samples=n_samples,
            scale=scale,
            logdir=logdir,
            log_vars=[])
        tf.global_variables_initializer().run()


    sess = ed.get_session()


    # Add node of posterior to graph
    prediction_post = ed.copy(prediction,
                              { w_h: qw_h, b_h: qb_h,
                                w_a: qw_a, b_a: qb_a })

    #with tf.contrib.tfprof.ProfileContext('../dat/train_dir') as pctx:
    time_start = time.time()
    time_start_epoch = time_start
    print('Parameter `n_cats`: {0}'.format(n_cats))

    for i in range(n_iter):

        x_batch, y_batch, y_error_batch = next(batch_generator)
        feed_dict = {x: x_batch,
                        y_ph: y_batch,
                        y_error: y_error_batch}

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        many_runs_timeline = TimeLiner()
        with Timer():
            _, t, loss = sess.run(
                [ inference.train, inference.increment_t,
                inference.loss ],
                feed_dict,
                options=run_options,
                run_metadata=run_metadata)
        inference.train_writer.add_run_metadata(run_metadata, 'step%d' % i)

        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        many_runs_timeline.update_timeline(chrome_trace)

        # Validation for each epoch
        if (i+1) % mnist_.n_batches_per_epoch == 0:

            epoch = int( (i+1) / mnist_.n_batches_per_epoch )
            print('\nFinished the {0}-th epoch'.format(epoch))
            print('Elapsed time {0} sec.'.format(time.time()-time_start))

            # Get validation data
            x_valid, y_valid, y_error_valid = mnist_.validation_data
            x_valid, y_valid, y_error_valid = \
                shuffle(x_valid, y_valid, y_error_valid)
            x_valid, y_valid, y_error_valid = \
                x_valid[:128], y_valid[:128], y_error_valid[:128]

            # Get accuracy
            n_models = 100  # number of Monte Carlo neural network models.
            # shape: [n_models, n_test_data, n_outputs]
            softmax_vals = [prediction_post.eval(feed_dict={x: x_valid})
                            for i in range(n_models)]
            # shape: [n_test_data, n_outputs]
            mean_softmax_vals = np.mean(softmax_vals, axis=0)
            # shape: [n_test_data]
            y_pred = np.argmax(mean_softmax_vals, axis=-1)
            accuracy = get_accuracy(y_pred, y_valid)

            print('Accuracy on validation data: {0} %'\
                    .format(accuracy/mnist_.batch_size*100))
            time_start = time.time()  # re-initialize.

    many_runs_timeline.save(
        '../dat/timelines/timeline_with_{0}_cats.json'\
        .format(n_cats))
    time_end = time.time()
    print('------ Elapsed {0} sec in training.\n\n'\
          .format(time_end-time_start))


    '''
    # EVALUATE
    x_test, y_test, y_error_test = mnist_.test_data
    n_test_data = len(y_test)
    print('{0} test data.'.format(n_test_data))

    n_models = 500  # number of Monte Carlo neural network models.
    # shape: [n_models, n_test_data, n_outputs]
    prediction_vals = [prediction_post.eval(feed_dict={x: x_test})
                        for i in range(n_models)]

    # Get accuracy
    n_models = 100  # number of Monte Carlo neural network models.
    # shape: [n_models, n_test_data, n_outputs]
    softmax_vals = [prediction_post.eval(feed_dict={x: x_test})
                    for i in range(n_models)]
    # shape: [n_test_data, n_outputs]
    mean_softmax_vals = np.mean(softmax_vals, axis=0)
    # shape: [n_test_data]
    y_pred = np.argmax(mean_softmax_vals, axis=-1)
    accuracy = get_accuracy(y_pred, y_test)

    print('Accuracy on test data: {0} %'\
            .format(accuracy/mnist_.batch_size*100))
    '''

    ## Save the training result
    #pretrained = get_variable_value_dict(sess)
    #print(pretrained.keys())
    #try:
    #    pickle.dump(pretrained, open(path_to_pretrained, 'wb'))
    #except Exception as e:
    #    print('Fail in saving trained variable to disk - ', e)



def main(n_cats):

    if n_cats == 0:
        without_cat()

    elif n_cats > 0:
        with_cat(n_cats)



if __name__ == '__main__':

    import sys
    n_cats = int(sys.argv[1])
    main(n_cats)
