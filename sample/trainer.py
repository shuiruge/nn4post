#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
A trainer that implememts the material in chapter 11 of the book _Hands-On \
Machine Learning with Scikit-Learn and TensorFlow_.

This trainer will implement:

  1. `tensorboard` logging;
  2. saving checkpoints;
  3. retoring checkpoints;
  4. gradient clipping;
  5. learning-rate scheduling


Motivation
----------
I try to use the `Trainer` provided by tflearn. But it is hard to use.
"""


import abc
import numpy as np
import tensorflow as tf



def iterate(sess, train_op, feed_dict_generator,
            summarizer=None, writer=None, global_step=None,
            options=None, run_metadata=None):
  """Iterates one step for optimizing the `train_op`.

  CAUTION:
    This "function" will change the state of the `sess` and the `global_step`
    (if not `None`).

  NOTE:
    This implementation abstracts all, and nothing else is essential. (That is,
    all args in all employed functions (methods) have been fullfilled.)

  Args:
    sess:
      An instance of `tf.Session()`, as the session this iteration works on.

    train_op:
      `Op`, as the train-op to be iterated. Ensure that it has been initialized.

    feed_dict_generator:
      Generator that emits a `feed_dict` associated to the `tf.placeholder`s
      needed by the `train_op`, at each calling of `next()`.

    summarizer:
      A "summary op" that summarizes the graph, e.g. `tf.summary.merge_all`,
      optional.

    writer:
      An instance of `tf.summary.FileWriter` that writes the summary into disk.
      If the `summarizer` is `None` (as default), then this argument is useless,
      optional.

    global_step:
      An un-trainalbe variable with a scalar shape and an integer dtype,
      optional.

    options:
      A `[RunOptions]` protocol buffer or `None`, as the associated argument of
      `tf.Session.run()`, optional.

    run_metadata:
      A `[RunMetadata]` protocol buffer or `None`, as the associated argument of
      `tf.Session.run()`, optional.

  Returns:
    `True` if succeed in iterating, and `False` if failed by a `StopIteration`
    exception in calling `next(feed_dict_generator)`.
  """

  fetches = [train_op]
  if summarizer is not None:
    fetches.append(summarizer)

  try:
    feed_dict = next(feed_dict_generator)
  except StopIteration:
    return False

  fetch_vals = sess.run(fetches
                        feed_dict=feed_dict,
                        options=options,
                        run_metadata=run_metadata)

  if summarizer is not None:
    _, summary = fetch_vals

    if global_step is not None:
      sess.run(global_step.assign_add(1))

    writer.add_summary(summary, global_step=global_step)

  return True



class BaseTrainer(object):
  """Abstract base class of trainer that implememts the material in chapter 11
  of the book _Hands-On Machine Learning with Scikit-Learn and TensorFlow_.

  Args:
    log_vars:
      List of instances of `tf.Variable`, as the variables to be logged into
      TensorBoard.
  """

  def __init__(self, loss, logdir=None, dir_to_save=None,
               graph=None, sess=None):

    self.loss = loss
    self.logdir = logdir
    self.dir_to_save = dir_to_save

    self.optimizer = self.get_optimizer()
    self.train_op = self.build_optimization()
    self.summarizer = self.build_summarization()
    self.graph = graph if graph is not None else tf.get_default_graph()
    self.sess = sess if sess is not None else tf.Session()


  @abc.abstractmethod
  def get_optimizer(self):
    pass


  @abc.abstractmethod
  def get_grad_and_var_list(self):
    pass


  def build_optimization(self):
    """Implements the scope `optimization`.

    Returns:
      Op for optimization in one iteration.
    """

    with self.graph.as_default():

      with tf.name_scope('optimization'):

        gvs = self.get_grad_and_var_list()
        train_op = self.optimizer.apply_gradients(gvs)

    return train_op


  @abc.abstractmethod
  def build_summarization(self, *args, **kwargs):
    """Implements the scope `summarization`.

    Returns:
      Op for summarization (to `tensorboard`) in one iteration.
    """

    with self.graph.as_default():

      with tf.name_scope('summarization'):

        summarizer = None

    return summarizer


  @abc.abstractmethod
  def save(self):
    pass


  @abc.abstractmethod
  def restore(self):
    pass


  def fit(self, n_iters, feed_dict_generator, skip_step=100):

    self.sess.run(self.initializer)

    self.restore()

    for i in range(n_iters):

      iterate(self.sess, self.train_op, feed_dict_generator,
              summarizer=self.summarizer, writer=self.writer,
              global_step=self.global_step)

      if (i+1) % skip_step:

        self.save()
