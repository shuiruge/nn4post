#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
This is a rough version of nn4post with mean-field approximation (so that the
model can be transfered).

If it works well, then re-arrange it to inherit `edward.Inference`.


Documentation
-------------
C.f. '../docs/nn4post.pdf'.



Efficiency
----------

In [link](https://www.tensorflow.org/api_docs/python/tf/contrib/bayesflow/\
entropy/elbo_ratio):

> The term E_q[ Log[p(Z)] ] is always computed as a sample mean. The term
> E_q[ Log[q(z)] ] can be computed with samples, or an exact formula if
> q.entropy() is defined. This is controlled with the kwarg form.

Now it is clear. The surprising efficiency of TensorFlow in calculating
`elbo` is gained by and only by the usage of analytic entropy called by
`q.entropy()`. That mystery encountered in Shanghai.

However, `tf.distributions.Mixture` has no `entropy` method. Instead, use
method `entropy_lower_bound`. This is plausible, c.f. "Derivation" section.

So, all left is how to efficiently compute the **value** of likelihood on
the **sampled values** of parameters.


Derivation
----------

Denote $z$ element in parameter-space, $D$ data, $c_i$ the category
distribution logits, and $q_i$ the component distribution. We have

$$ \textrm{KL} ( q \mid p ) = - \textrm{ELBO} + \ln p(D), $$

wherein

$$ \textrm{ELBO} := H[q] + E_q [ \ln p(z, D) ] $$

as usual, thus

$$ \textrm{ELBO} \leq \ln p(D). $$

With `entropy_lower_bound` instead of `entropy`, and let

$$ \mathcal{E} := \sum_i c_i H[q_i] + E_q [ \ln p(z, D) ], $$

we have, since $ \sum_i c_i H[q_i] \leq H[q] $,

$$ \mathcal{E} \leq \textrm{ELBO} \leq \ln p(D). $$

Thus, if let $ \textrm{loss} := - \mathcal{E} $, optimizer can find the
minimum of the loss-function, as expected.


### On Edward

It seems that `edward.util.copy` is costy, and seems non-essential. Thus
we shall avoid employing it.


### TODO
Find that the memory usage varies greatly while training. Try to find out the
reason.
"""


import os
import six
import tensorflow as tf
import numpy as np
# -- `contrib` module in TF 1.3
from tensorflow.contrib.distributions import \
    Categorical, NormalWithSoftplusScale
from tensorflow.contrib.bayesflow import entropy
# -- To be changed in TF 1.4
from mixture_same_family import MixtureSameFamily
from tools import ensure_directory, Timer, TimeLiner
import mnist
from edward.util import Progbar
import pickle
import time
from tensorflow.python.client import timeline



# For testing (and debugging)
tf.set_random_seed(42)
np.random.seed(42)


# --- Data ---

noise_std = 0.1
batch_size = 128  # test!
mnist_ = mnist.MNIST(noise_std, batch_size)
batch_generator = mnist_.batch_generator()



# --- Parameters ---

N_CATS = 10
N_SAMPLES = 100
SCALE = mnist_.n_data / mnist_.batch_size
LOG_DIR = '../dat/logs/'
DIR_TO_CKPT = '../dat/checkpoints'


# --- Setup Computational Graph ---

with tf.name_scope('model'):

    def MODEL(inputs, params):
        """ Shall be implemented by TensorFlow. This is an example, as a shallow
        neural network.

        Args:
            inputs:
                `dict`, like `{'x_1': x_1, 'x_2': x_2}, with values Tensors.
            params:
                `dict`, like `{'w': w, 'b': b}, with values Tensors.

        Returns:
            Tensor.
        """
        # shape: `[None, n_hiddens]`
        hidden = tf.sigmoid(
            tf.matmul(inputs['x'], params['w_h']) + params['b_h'])
        # shape: `[None, n_outputs]`
        activation = tf.nn.softmax(
            tf.matmul(hidden, params['w_a']) + params['b_a'])
        return activation



with tf.name_scope('prior'):

    n_inputs = 28 * 28  # number of input features.
    n_hiddens = 100  # number of perceptrons in the (single) hidden layer.
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
    latent_vars = {
        'w_h': w_h, 'w_a': w_a,
        'b_h': b_h, 'b_a': b_a,
    }



with tf.name_scope('data'):

    x = tf.placeholder(shape=[None, n_inputs], dtype=tf.float32, name='x')
    y = tf.placeholder(shape=[None, n_outputs], dtype=tf.float32, name='y')
    y_err = tf.placeholder(shape=y.shape, dtype=tf.float32, name='y_err')
    data = {'x': x, 'y': y, 'y_err': y_err}



with tf.name_scope('posterior'):

    with tf.name_scope('likelihood'):

        def chi_square(model_output):
            """ (Temporally) assume that the data obeys a normal distribution,
            realized by Gauss's limit-theorem. """

            normal = NormalWithSoftplusScale(loc=y, scale=y_err)
            un_scaled_val = tf.reduce_sum(normal.log_prob(model_output))
            if SCALE is None:
                return un_scaled_val
            else:
                return SCALE * un_scaled_val


        def log_likelihood(params):
            """
            Args:
                params: The same argument in the `MODEL`.
            Returns:
                Scalar.
            """
            return chi_square(MODEL(inputs={'x': x}, params=params))


    with tf.name_scope('prior'):

        def log_prior(params):
            """
            Args:
                params: The same argument in the `MODEL`.
            Returns:
                Scalar.
            """

            log_priors = [qz.log_prob(params[z]) for z, qz
                          in latent_vars.items()]
            total_log_prior = tf.reduce_sum(
                [tf.reduce_sum(_) for _ in log_priors]
            )
            return total_log_prior


    param_names_in_order = sorted(latent_vars.keys())
    param_shapes = [latent_vars[z].batch_shape.as_list()
                    for z in param_names_in_order]
    param_sizes = [np.prod(param_shape) for param_shape in param_shapes]
    param_space_dim = sum(param_sizes)
    print(' --- Parameter-space Dimension: {0}'.format(param_space_dim))

    def parse_params(theta):
        """
        Args:
            theta:
                Tensor with shape `[param_space_dim]`, as one element in the
                parameter-space, obtained by flattening the `params` in the
                arguments of the `MODEL`, and then concatenating by the order
                of the `param_names_in_order`.
        Returns:
            `dict` with keys the same as `latent_vars`, and values Tensors with
            shape the same as the values of `latent_vars`.
        """
        splited = tf.split(theta, param_sizes)
        reshaped = [tf.reshape(flat, shape) for flat, shape
                    in list(zip(splited, param_shapes))]
        return {z: reshaped[i] for i, z in enumerate(param_names_in_order)}


    def log_posterior(theta):
        """
        Args:
            theta: Tensor with shape `[param_space_dim]`.
        Returns:
            Scalar.
        """
        params = parse_params(theta)
        return log_likelihood(params) + log_prior(params)



with tf.name_scope('inference'):

      with tf.name_scope('variables'):

          with tf.name_scope('initial_values'):
            init_cat_logits = tf.zeros([N_CATS])
            init_loc = tf.random_normal([param_space_dim, N_CATS])
            init_softplus_scale = tf.zeros([param_space_dim, N_CATS])
            # Or using the values in the previous calling of this script.

          cat_logits = tf.Variable(
              init_cat_logits,
              name='cat_logits')
          loc = tf.Variable(
              init_loc,
              name='loc')
          softplus_scale = tf.Variable(
              init_softplus_scale,
              name='softplus_scale')

      with tf.name_scope('q_distribution'):

          mixture_distribution = Categorical(logits=cat_logits)

          components_distribution = \
              NormalWithSoftplusScale(loc=loc, scale=softplus_scale)

          q = MixtureSameFamily(mixture_distribution,
                                components_distribution)



with tf.name_scope('loss'):

    with tf.name_scope('param_samples'):
        # shape: `[N_SAMPLES, param_space_dim]`
        thetas = q.sample(N_SAMPLES)

    # shape: `[N_SAMPLES]`
    with tf.name_scope('log_p'):
        log_ps = tf.map_fn(log_posterior, thetas, name='log_ps')
        log_p_mean = tf.reduce_mean(log_ps, name='log_p_mean')

    with tf.name_scope('q_entropy'):
        # Get `q_entropy`
        # C.f. [here](http://www.biopsychology.org/norwich/isp/chap8.pdf).
        with tf.name_scope('cat_weights'):
            cat_weights = tf.nn.softmax(cat_logits)
        with tf.name_scope('gauss_entropies'):
            gauss_entropies = (
                0.5 * np.log(2. * np.pi * np.e)
                + tf.log(tf.nn.softplus(softplus_scale))
            )
        with tf.name_scope('entropy_lower_bound'):
            entropy_lower_bound =  tf.reduce_sum(cat_weights * gauss_entropies)
        q_entropy = entropy_lower_bound  # as the approximation.

    loss = - ( log_p_mean + q_entropy )


    ''' Or use TF implementation
    def log_p(thetas):
        return tf.map_fn(log_posterior, thetas)

    elbos = entropy.elbo_ratio(log_p, q, z=thetas)
    loss = - tf.reduce_mean(elbos)
    # NOTE:
    #   TF uses direct Monte Carlo integral to compute the `q_entropy`,
    #   thus is quite slow.
    '''



with tf.name_scope('optimization'):

    #optimizer = tf.train.RMSPropOptimizer
    optimizer = tf.train.AdamOptimizer
    learning_rate = 0.01
    optimize = optimizer(learning_rate).minimize(loss)



with tf.name_scope('auxiliary_ops'):

    with tf.name_scope('summarizer'):

        with tf.name_scope('loss'):
            tf.summary.scalar('loss', loss)
            tf.summary.histogram('loss', loss)

        ## It seems that, up to TF version 1.3,
        ## `tensor_summary` is still under building
        #with tf.name_scope('variables'):
        #    tf.summary.tensor_summary('cat_logits', cat_logits)
        #    tf.summary.tensor_summary('loc', loc)
        #    tf.summary.tensor_summary('softplus_scale', softplus_scale)

        # -- And Merge them All
        summary = tf.summary.merge_all()





# --- Training ---

time_start = time.time()
logdir = '../dat/logs'
writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    n_epochs = 1
    #n_iter = mnist_.n_batches_per_epoch * n_epochs
    n_iter = 3  # test!
    progbar = Progbar(n_iter)

    for i in range(n_iter):

        x_batch, y_batch, y_err_batch = next(batch_generator)
        feed_dict = {x: x_batch,
                        y: y_batch,
                        y_err: y_err_batch}

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        many_runs_timeline = TimeLiner()

        _, loss_val, summary_val = sess.run(
            [optimize, loss, summary],
            feed_dict,
            options=run_options,
            run_metadata=run_metadata,
        )
        writer.add_run_metadata(run_metadata, 'step%d' % i)
        writer.add_summary(summary_val, global_step=i)
        progbar.update(i, {'Loss': loss_val})

        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        many_runs_timeline.update_timeline(chrome_trace)

        '''
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
            softmax_vals = [XXX.eval(feed_dict={x: x_valid})
                            for i in range(n_models)]
            # shape: [n_test_data, n_outputs]
            mean_softmax_vals = np.mean(softmax_vals, axis=0)
            # shape: [n_test_data]
            y_pred = np.argmax(mean_softmax_vals, axis=-1)
            accuracy = get_accuracy(y_pred, y_valid)

            print('Accuracy on validation data: {0} %'\
                    .format(accuracy/mnist_.batch_size*100))
        '''

    many_runs_timeline.save('../dat/timelines/timeline.json')
    time_end = time.time()
    print(' --- Elapsed {0} sec in training'.format(time_end-time_start))


    variable_vals = {
        'cat_logits': cat_logits.eval(),
        'loc': loc.eval(),
        'softplus_scale': softplus_scale.eval()
    }


'''Profiling

By checking TensorBoard and timeline (by chrome://tracing), we found that the
most temporally costy is the "Mul" ops in multiplications between the standard
Gaussian samples and scales, and between the Gaussian samples and categorical
samples. So, we expect that GPU can make this script quite faster.

The memory costs come from anywhere that a large tensor with shape
`[N_SAMPLES, param_space_dim]` is generate.
'''
