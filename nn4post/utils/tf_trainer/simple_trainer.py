#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script test `BaseTrainer` by inherition.

XXX
C.f. https://www.tensorflow.org/get_started/mnist/beginners.
"""


import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from .base_trainer import BaseTrainer
from .utils import ensure_directory



def create_sess(graph, sess_config, sess_target, debug):

  sess = tf.Session(graph=graph, config=sess_config, target=sess_target)

  if debug:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

  return sess


def create_train_ops(graph, loss, gvs, optimizer):

  with graph.as_default():

    with tf.name_scope('optimization'):

      if gvs is not None:
        train_op = optimizer.apply_gradients(gvs)
      else:
        # The argument `global_step` in `minimize()` will automatically
        # `assign_add(1)` to what is put into the arugment, thus shall not be
        # filled, keep it `None` (as defalut)
        train_op = optimizer.minimize(loss, name='train_op')

  return [train_op]


def create_summarizer(graph, log_vars):
  """
  NOTE:
    Generally, one summarizer is enough. When you needs two, e.g. one for
    summarizing training process and the other validation process, you can
    just use one summary_op like `tf.summary.scalar(name, value)`, with
    `name` a placeholder of stringy type. and feed in `feed_dict_generator`.
  """

  if log_vars is None:

    summarizer = None

  else:

    with graph.as_default():

      with tf.name_scope('summarization'):

        for v in log_vars:
          if v.shape == tf.TensorShape([]):
            # `v` is a scalar.
            tf.summary.scalar(v.name, v)
          else:
            # `v` is a tensor.
            tf.summary.tensor_summary(v.name, v)
          tf.summary.histogram(v.name, v)

        summarizer = tf.summary.merge_all()

  return summarizer


def create_writer(graph, logdir):
  if logdir is None:
    writer = None
  else:
    writer = tf.summary.FileWriter(logdir, graph)
  return writer


def create_saver(graph):
  # Saver shall be initialized within the graph
  with graph.as_default():
    saver = tf.train.Saver()
  return saver


def save(saver, sess, dir_to_ckpt, global_step):
  saver.save(
      sess,
      os.path.join(dir_to_ckpt, 'checkpoint'),
      global_step=global_step)


def restore(dir_to_ckpt, saver, sess):

  # Get checkpoint
  # CAUTION that the arg of `get_checkpoint_state` is
  # `checkpoint_dir`, i.e. the directory of the `checkpoint`
  # to be restored from.
  ckpt = tf.train.get_checkpoint_state(dir_to_ckpt)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('INFO - Restored from {}.'.format(dir_to_ckpt))
    return True
  else:
    print("INFO - There's been no ckpt yet.")
    return False



class SimpleTrainer(BaseTrainer):
  """With the basic implementation for each method."""

  def __init__(self, graph=None, loss=None, gvs=None, optimizer=None,
               logdir=None, log_vars=None, writer_skip_step=10,
               dir_to_ckpt=None, debug=False,  *args, **kwargs):

    self.graph = tf.get_default_graph() if graph is None \
                 else graph

    # For optimization
    self.loss = loss
    self.gvs = gvs
    self.optimizer = tf.train.AdamOptimizer(0.01) if optimizer is None \
                     else optimizer
    self.train_ops = create_train_ops(
        self.graph, self.loss, self.gvs, self.optimizer)

    # For TensorBoard
    self.logdir = logdir
    self.log_vars = [self.loss] if log_vars is None \
                    else log_vars
    self.writer_skip_step = writer_skip_step
    self.summarizer = create_summarizer(self.graph, self.log_vars)
    self.writer = create_writer(self.graph, self.logdir)

    # For saving checkpoint
    self.dir_to_ckpt = dir_to_ckpt
    if self.dir_to_ckpt is not None:
      ensure_directory(self.dir_to_ckpt)

    # For debug
    # Needed in `self.create_sess()`, thus before calling `super()`
    self.debug = debug

    # Recall that `super()` can only be called in the end. (C.f. the docsting of
    # `BaseTrainer`.)
    super().__init__(graph=self.graph, *args, **kwargs)


  def create_saver(self):
    saver = create_saver(self.graph)
    return saver


  def create_sess(self):
    sess = create_sess(self.graph, self.sess_config,
                       self.sess_target, self.debug)
    return sess


  def get_iter_ops(self):
    return self.train_ops


  def get_summarizer(self):
    if self.get_global_step_val() % self.writer_skip_step == 0:
      return self.summarizer
    else:
      return None


  def get_writer(self):
    if self.get_global_step_val() % self.writer_skip_step == 0:
      return self.writer
    else:
      return None


  def save(self):
    if self.dir_to_ckpt is not None:
      save(self.saver, self.sess, self.dir_to_ckpt, self.global_step)


  def restore(self):
    if self.dir_to_ckpt is None:
      return None
    else:
      return restore(self.dir_to_ckpt, self.saver, self.sess)
