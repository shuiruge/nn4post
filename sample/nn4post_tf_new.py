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
import tensorflow as tf
import numpy as np
# -- `contrib` module in TF 1.3
from tensorflow.contrib.distributions import (
    Categorical, NormalWithSoftplusScale,
    MultivariateNormalDiagWithSoftplusScale, Mixture
)
from tensorflow.contrib.bayesflow import entropy
# -- To be changed in TF 1.4
from mixture_same_family import MixtureSameFamily
from tools import ensure_directory, Timer, TimeLiner
import mnist
import pickle
import time
from tensorflow.python.client import timeline
from independent import Independent



# For testing (and debugging)
tf.set_random_seed(42)
np.random.seed(42)


def build_graph(model, latent_vars, input_data, output_data,
                likelihood=None, optimizer=None, base_graph=None,
                log_accurate_loss=False):
  """
  Args:
    model:
      Callable from `dict` `input_data` and `dict` `latent_vars` to `dict`
      `output_data`.

    latent_vars:
      `dict`, with keys `str`s and values `tf.Tensor`s.

    input_data:
      `dict`, with keys `str`s and values `tf.Tensor`s.

    output_data:
      `dict`, with keys `str`s and each value a tuple of two `tf.Tensor`s, for
      observed value and the observed error of the associated output within the
      model's outputs (maybe non-unique).

    likelihood:
      `None` or callable from `dict` `latent_vars` to scalar, optional.

    optimizer:
      An instance of optimizers that inherites the `tf.train.Optimizer`.

    base_graph:
      An instance of `tf.Graph` or `None`, upon which our computational graph is
      built, optional. If `None`, then our computational graph is built on the
      graph `tf.get_default_graph()`.

    log_accurate_loss:
      `bool`, optional. If `True`, then accurate loss, computed by full Monte-
      Carlo integral, will be computed and then logged in TensorBoard.


  Returns:
    A tuple of two elements. The first is the built computational graph, and the
    later is a `dict` of `dict`s of ops, with the structure:
    ```python

        ops = {
            'vars': {
                'cat_logits': cat_logits,
                'loc': loc,
                'softplus_scale': softplus_scale,
            },
            'feed': {
                'learning_rate': learning_rate,
                'n_samples': n_samples,
                'scale': scale,
            },
            'loss': {
                'approximate_loss': approximate_loss,
            },
            'train': {
                'train_op': train_op,
                'summary_op': summary_op,
            },
        }
    ```
    as a convienent representation of the collections of the computational graph.


  Example:
    ```python

    n_inputs = 28 * 28  # number of input features.
    n_outputs = 10  # number of perceptrons in the output layer.

    x = tf.placeholder(shape=[None, n_inputs], dtype=tf.float32, name='x')
    y = tf.placeholder(shape=[None, n_outputs], dtype=tf.float32, name='y')
    y_err = tf.placeholder(shape=y.shape, dtype=tf.float32, name='y_err')

    input_data = {
        'x': x,
    }
    output_data = {
        'y': (y, y_err),
    }

    def model(inputs, params):
        '''Shall be implemented by TensorFlow. This is an example, as a logits-
        regression.

        Args:
            inputs:
                `dict`, like `{'x_1': x_1, 'x_2': x_2}, with values Tensors.
            params:
                `dict`, like `{'w': w, 'b': b}, with values Tensors.

        Returns:
            `dict`, like `{'y': y}`.
        '''
        # shape: `[None, n_outputs]`
        y = tf.sigmoid(
            tf.matmul(inputs['x'], params['w']) + params['b'])
        return {'y': y}

    w = NormalWithSoftplusScale(
        loc=tf.zeros([n_inputs, n_outputs]),
        scale=tf.ones([n_inputs, n_outputs]),
        name='w')
    b = NormalWithSoftplusScale(
        loc=tf.zeros([n_outputs]),
        scale=tf.ones([n_outputs]) * 100,
        name='b_h')

    latent_vars = {
        'w': w,
        'b': b,
    }

    graph, ops = build_graph(model, latent_vars, input_data, output_data)
  ```
  """

  graph = tf.get_default_graph() if base_graph is None else base_graph

  with graph.as_default():

    with tf.name_scope('likelihood'):

      if likelihood is None:

        scale = tf.placeholder(shape=[], dtype=tf.float32, name='scale')

        def chi_square(model_outputs):
          """ (Temporally) assume that the data obeys a normal distribution,
          realized by Gauss's limit-theorem.

          Args:
            model_outputs: `dict`.
          Returns:
            Scalar.
          """
          un_scaled_val = 0.0
          for y in output_data.keys():
            y_val, y_err = output_data[y]
            normal = NormalWithSoftplusScale(loc=y_val, scale=y_err)
            un_scaled_val += tf.reduce_sum(normal.log_prob(model_outputs[y]))
          return scale * un_scaled_val


        def log_likelihood(params):
          """
          Args:
            params: The same argument in the `model`.
          Returns:
            Scalar.
          """
          return chi_square(model(inputs=input_data, params=params))


    with tf.name_scope('prior'):

      def log_prior(params):
        """
        Args:
          params: The same argument in the `model`.
        Returns:
          Scalar.
        """

        log_priors = [qz.log_prob(params[z])
                      for z, qz in latent_vars.items()]
        total_log_prior = tf.reduce_sum(
            [tf.reduce_sum(_) for _ in log_priors]
        )
        return total_log_prior


    with tf.name_scope('posterior'):

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
            arguments of the `model`, and then concatenating by the order
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

        # NOTE:
        #   Using `Mixture` + `NormalWithSoftplusScale` + `Independent` out-
        #   performs to 1) using `MixtureSameFamily` and 2) `Mixture` +
        #   `MultivariateNormalDiagWithSoftplusScale` on both timming and
        #   memory profiling. The (1) is very memory costy.
        cat = Categorical(logits=cat_logits)
        locs = tf.unstack(loc, axis=1)
        softplus_scales = tf.unstack(softplus_scale, axis=1)
        components = [
            Independent(
                NormalWithSoftplusScale(locs[i], softplus_scales[i])
            ) for i in range(N_CATS)
        ]
        q = Mixture(cat, components)


    with tf.name_scope('loss'):

      n_samples = tf.placeholder(shape=[], dtype=tf.int32, name='n_samples')

      with tf.name_scope('param_samples'):
        # shape: `[n_samples, param_space_dim]`
        thetas = q.sample(n_samples)

      # shape: `[n_samples]`
      with tf.name_scope('log_p'):

        def log_p(thetas):
          """Vectorize `log_posterior`."""
          return tf.map_fn(log_posterior, thetas)

        log_p_mean = tf.reduce_mean(log_p(thetas), name='log_p_mean')

      with tf.name_scope('q_entropy'):
        q_entropy = q.entropy_lower_bound()

      with tf.name_scope('approximate_loss'):
        approximate_loss = - ( log_p_mean + q_entropy )

      if log_accurate_loss:
        with tf.name_scope('accurate_loss'):
          accurate_loss = - entropy.elbo_ratio(log_p, q, z=thetas)


    with tf.name_scope('optimization'):

      if optimizer is None:
        optimizer = tf.contrib.opt.NadamOptimizer
      learning_rate = tf.placeholder(shape=[], dtype=tf.float32,
                                     name='learning_rate')
      train_op = optimizer(learning_rate).minimize(approximate_loss)
      #train_op = optimizer(learning_rate).minimize(accurate_loss)  # test!



    with tf.name_scope('auxiliary_ops'):

      with tf.name_scope('summarizer'):

        with tf.name_scope('approximate_loss'):
          tf.summary.scalar('approximate_loss', approximate_loss)
          tf.summary.histogram('approximate_loss', approximate_loss)

        if log_accurate_loss:
          with tf.name_scope('accurate_loss'):
            tf.summary.scalar('accurate_loss', accurate_loss)
            tf.summary.histogram('accurate_loss', accurate_loss)

        ## It seems that, up to TF version 1.3,
        ## `tensor_summary` is still under building
        #with tf.name_scope('variables'):
        #    tf.summary.tensor_summary('cat_logits', cat_logits)
        #    tf.summary.tensor_summary('loc', loc)
        #    tf.summary.tensor_summary('softplus_scale', softplus_scale)

        # -- And Merge them All
        summary_op = tf.summary.merge_all()

  # -- Collections
  ops = {
      'vars': {
          'cat_logits': cat_logits,
          'loc': loc,
          'softplus_scale': softplus_scale,
      },
      'feed': {
          'learning_rate': learning_rate,
          'n_samples': n_samples,
          'scale': scale,
      },
      'loss': {
          'approximate_loss': approximate_loss,
      },
      'train': {
          'train_op': train_op,
          'summary_op': summary_op,
      },
  }
  if log_accurate_loss:
    ops['loss']['accurate_loss'] = accurate_loss
  for class_name, op_dict in ops.items():
    for op_name, op in op_dict.items():
      graph.add_to_collection(op_name, op)

  return (graph, ops)



if __name__ == '__main__':

    """Test."""

    # --- Data ---

    noise_std = 0.1
    batch_size = 128
    mnist_ = mnist.MNIST(noise_std, batch_size)
    batch_generator = mnist_.batch_generator()


    # --- Parameters ---

    N_CATS = 10
    N_SAMPLES = 100
    SCALE = mnist_.n_data / mnist_.batch_size
    LOG_DIR = '../dat/logs/'
    DIR_TO_CKPT = '../dat/checkpoints'
    LOG_ACCURATE_LOSS = True
    PROFILING = False
    DEBUG = False


    # --- Setup Computational Graph ---

    n_inputs = 28 * 28  # number of input features.
    n_hiddens = 10  # number of perceptrons in the (single) hidden layer.
    n_outputs = 10  # number of perceptrons in the output layer.

    x = tf.placeholder(shape=[None, n_inputs], dtype=tf.float32, name='x')
    input_data = {
        'x': x,
    }

    y = tf.placeholder(shape=[None, n_outputs], dtype=tf.float32, name='y')
    y_err = tf.placeholder(shape=y.shape, dtype=tf.float32, name='y_err')
    output_data = {
        'y': (y, y_err),
    }

    def model(inputs, params):
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
        return {'y': activation}

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

    graph, ops = build_graph(model, latent_vars, input_data, output_data)



    # --- Training ---

    writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

    sess = tf.Session(graph=graph)

    if DEBUG:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    with sess:

        sess.run(tf.global_variables_initializer())

        n_epochs = 1
        #n_iter = mnist_.n_batches_per_epoch * n_epochs
        n_iter = 3  # test!

        for i in range(n_iter):

            x_batch, y_batch, y_err_batch = next(batch_generator)
            feed_ops = ops['feed']
            feed_dict = {
                x: x_batch,
                y: y_batch,
                y_err: y_err_batch,
                feed_ops['learning_rate']: 0.01,
                feed_ops['scale']: SCALE,
                feed_ops['n_samples']: N_SAMPLES,
            }

            train_ops = ops['train']

            if PROFILING:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                many_runs_timeline = TimeLiner()

                _, summary_val = sess.run(
                    [ train_ops['train_op'],
                      train_ops['summary_op'] ],
                    feed_dict,
                    options=run_options,
                    run_metadata=run_metadata
                )
                writer.add_run_metadata(run_metadata, 'step%d' % (i+1))
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                many_runs_timeline.update_timeline(chrome_trace)

            else:
                _, summary_val = sess.run(
                    [ train_ops['train_op'],
                      train_ops['summary_op'] ],
                    feed_dict
                )

            writer.add_summary(summary_val, global_step=i+1)

            '''
            # Validation for each epoch
            if (i+1) % mnist_.n_batches_per_epoch == 0:

                epoch = int( (i+1) / mnist_.n_batches_per_epoch )
                print('\nFinished the {0}-th epoch'.format(epoch))

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

        if PROFILING:
            many_runs_timeline.save('../dat/timelines/timeline.json')

        variable_vals = {
            'cat_logits': ops['vars']['cat_logits'].eval(),
            'loc': ops['vars']['loc'].eval(),
            'softplus_scale': ops['vars']['softplus_scale'].eval(),
        }

        with open('../dat/vars.pkl', 'wb') as f:
            pickle.dump(variable_vals, f)



    '''Profiling

    By checking TensorBoard and timeline (by chrome://tracing), we found that the
    most temporally costy is the "Mul" ops in multiplications between the standard
    Gaussian samples and scales, and between the Gaussian samples and categorical
    samples. So, we expect that GPU can make this script quite faster.

    The memory costs come from anywhere that a large tensor with shape
    `[n_samples, param_space_dim]` is generate.

    Setting `USE_MIXTURE = True` costs less memory and timing.
    '''
