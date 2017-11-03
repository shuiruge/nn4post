#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------

Test nn4post, comparing with standard single-peak variational inference,
directly on Gaussian mixture model.
"""


import tensorflow as tf
import numpy as np
# -- `contrib` module in TF 1.3
from tensorflow.contrib.distributions import (
    Categorical, NormalWithSoftplusScale,
    MultivariateNormalDiagWithSoftplusScale, Mixture
)
from tensorflow.contrib.bayesflow import entropy
# -- To be changed in TF 1.4
import os
import sys
sys.path.append('../sample/')
from tools import ensure_directory, Timer, TimeLiner
import mnist
import pickle
import time
from tensorflow.python.client import timeline
from independent import Independent
from tensorflow.python import debug as tf_debug




# For testing (and debugging)
seed = 123
tf.set_random_seed(seed)
np.random.seed(seed)



# --- Parameters ---

N_CATS = 5  # shall be varied.
N_SAMPLES = 100
PARAM_SPACE_DIM = 2
TARGET_N_CATS = 3  # shall be fixed.
LOG_ACCURATE_LOSS = True
PROFILING = False
DEBUG = False
SKIP_STEP = 20
#LOG_DIR = '../dat/logs/gaussian_mixture_model/{0}_{1}'\
#          .format(TARGET_N_CATS, N_CATS)
#DIR_TO_CKPT = '../dat/checkpoints/gaussian_mixture_model/{0}_{1}'\
#              .format(TARGET_N_CATS, N_CATS)
LOG_DIR = None
DIR_TO_CKPT = None
if DIR_TO_CKPT is not None:
    ensure_directory(DIR_TO_CKPT)



# --- Setup Computational Graph ---



with tf.name_scope('posterior'):

    target_cat_logits = tf.constant([-1., 0., 1.])  # shall be equally weighted.
    target_loc = tf.stack(
        [ tf.ones([PARAM_SPACE_DIM]) * (i - 1) * 3 for i in range(TARGET_N_CATS) ],
        axis=1)
    target_softplus_scale = tf.zeros([PARAM_SPACE_DIM, TARGET_N_CATS])

    p = Mixture(
        Categorical(logits=target_cat_logits),
        [ Independent(
             NormalWithSoftplusScale(target_loc[:,i], target_softplus_scale[:,i])
          ) for i in range(TARGET_N_CATS) ]
    )

    def log_posterior(theta):
        return p.log_prob(theta)



with tf.name_scope('inference'):

    with tf.name_scope('variables'):

        with tf.name_scope('initial_values'):

            init_cat_logits = tf.zeros([N_CATS])
            init_locs = [tf.random_normal([PARAM_SPACE_DIM]) * 5.0
                        for i in range(N_CATS) ]
            init_softplus_scales = [tf.ones([PARAM_SPACE_DIM]) * (1.0)
                                   for i in range(N_CATS)]
            # Or using the values in the previous calling of this script.

            cat_logits = tf.Variable(
                init_cat_logits,
                name='cat_logits')
            locs = [tf.ones([PARAM_SPACE_DIM]) * 3.] \
                 + [ tf.Variable(init_locs[i], name='loc_{}'.format(i))
                     for i in range(1, N_CATS) ]
            softplus_scales = [tf.zeros([PARAM_SPACE_DIM])] \
                            + [ tf.Variable(init_softplus_scales[i],
                                            name='softplus_scale_{}'.format(i))
                                for i in range(1, N_CATS) ]

    with tf.name_scope('q_distribution'):

        n_samples = tf.placeholder(shape=[], dtype=tf.int32, name='n_samples')

        # NOTE:
        #   Using `Mixture` + `NormalWithSoftplusScale` + `Independent` out-
        #   performs to 1) using `MixtureSameFamily` and 2) `Mixture` +
        #   `MultivariateNormalDiagWithSoftplusScale` on both timming and
        #   memory profiling. The (1) is very memory costy.
        cat = Categorical(logits=cat_logits)
        components = [
            Independent(
                NormalWithSoftplusScale(locs[i], softplus_scales[i])
            ) for i in range(N_CATS)
        ]
        q = Mixture(cat, components)
        something = q.sample(n_samples)



with tf.name_scope('loss'):

    with tf.name_scope('param_samples'):
        # shape: `[n_samples, PARAM_SPACE_DIM]`
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
        approximate_loss = - ( log_p_mean + q_entropy ) / PARAM_SPACE_DIM # test!

    if LOG_ACCURATE_LOSS:
        with tf.name_scope('accurate_loss'):
            accurate_loss = - entropy.elbo_ratio(log_p, q, z=thetas) / PARAM_SPACE_DIM # test!


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

    optimizer = tf.train.RMSPropOptimizer
    #optimizer = tf.contrib.opt.NadamOptimizer
    #optimizer = tf.train.AdamOptimizer
    learning_rate = tf.placeholder(shape=[], dtype=tf.float32,
                                   name='learning_rate')

    optimize_approximate_loss = \
        optimizer(learning_rate).minimize(approximate_loss)
    optimize_accurate_loss = \
        optimizer(learning_rate).minimize(accurate_loss)
    #optimize = optimize_approximate_loss
    optimize = optimize_accurate_loss


with tf.name_scope('auxiliary_ops'):

    with tf.name_scope('summarizer'):

        with tf.name_scope('approximate_loss'):
            tf.summary.scalar('approximate_loss', approximate_loss)
            tf.summary.histogram('approximate_loss', approximate_loss)

        if LOG_ACCURATE_LOSS:
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
        summary = tf.summary.merge_all()





# --- Training ---

time_start = time.time()
if LOG_DIR:
    writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
if DIR_TO_CKPT:
    saver = tf.train.Saver()

sess = tf.Session()

if DEBUG:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

with sess:

    sess.run(tf.global_variables_initializer())

    # -- Resotre from checkpoint
    initial_step = 1
    if DIR_TO_CKPT is not None:
        ckpt = tf.train.get_checkpoint_state(DIR_TO_CKPT)
        if ckpt and ckpt.model_checkpoint_path:
            try:  # test!
                saver.restore(sess, ckpt.model_checkpoint_path)
                initial_step = int(ckpt.model_checkpoint_path\
                                .rsplit('-', 1)[1])
                print('Restored from checkpoint at global step {0}'\
                    .format(initial_step))
            except Exception as e:
                print('ERROR - {0}'.format(e))
                print('WARNING - Continue without restore.')
    step = initial_step

    print(target_cat_logits.eval())
    print(target_loc.eval())
    print(target_softplus_scale.eval())

    print(cat_logits.eval())
    print(np.array([_.eval() for _ in locs]))
    print(np.array([_.eval() for _ in softplus_scales]))


    n_iter = 10 ** 3

    for i in range(n_iter):

        step = initial_step + (i + 1)

        feed_dict = {
            learning_rate: 0.03,
            n_samples: N_SAMPLES,
        }

        if PROFILING:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            many_runs_timeline = TimeLiner()

            _, approximate_loss_val, summary_val = sess.run(
                [optimize, approximate_loss, summary],
                feed_dict,
                options=run_options,
                run_metadata=run_metadata
            )
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            many_runs_timeline.update_timeline(chrome_trace)

        if LOG_DIR:
            writer.add_run_metadata(run_metadata, 'step%d' % step)

        else:
            _, approximate_loss_val, summary_val = sess.run(
                [optimize, approximate_loss, summary],
                feed_dict
            )

        if LOG_DIR:
            writer.add_summary(summary_val, global_step=step)

        # Save checkpoint
        if DIR_TO_CKPT is not None:
            path_to_ckpt = os.path.join(DIR_TO_CKPT, 'checkpoint')
            if step % SKIP_STEP == 0:
                saver.save(sess, path_to_ckpt, global_step=step)

        if step % 100 == 0:
            print(step, approximate_loss_val)
            print(cat_logits.eval())
            print(np.array([_.eval() for _ in locs]))
            print(np.array([_.eval() for _ in softplus_scales]))


    if PROFILING:
        many_runs_timeline.save('../dat/timelines/timeline.json')

    time_end = time.time()
    print(' --- Elapsed {0} sec in training'.format(time_end-time_start))




'''Log

### TODO:

* varying `init_softplus_scale`.

* `NadamOptimizer`.

* the difference between `approximate_loss` and `accurate_loss`.


### Varying `init_softplus_scale`

Both work well.


### Varying cats

With 1K iterations:

- `TARGET_N_CATS = 1`, `N_CATS = 2`, minimal loss is abount `5.3`.

- `TARGET_N_CATS = 2`, `N_CATS = 2`, minimal loss is abount `5.3`.

- `TARGET_N_CATS = 2`, `N_CATS = 1`, minimal loss is abount XXX.

'''
