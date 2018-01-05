#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script involves `iterate` and `BaseTrainer`."""


import abc
import numpy as np
import tensorflow as tf
from tqdm import tqdm



def iterate(sess, iter_ops, feed_dict, summarizer=None, writer=None,
            global_step_val=None, options=None, run_metadata=None):
  """Iterates one step for the TensorFlow `Op`s in `iter_ops`.

  CAUTION:
    This "function" will change the state of the `sess`.

  NOTE:
    This implementation abstracts all, and nothing else is essential. (That is,
    all args in all employed functions (methods) have been fullfilled.)

    Since the saving process in TensorFlow is not achived by `Op` (as how the
    summarizing and writing to TensorBoard are done), it is not essential, thus
    will not also, be included herein.

  Args:
    sess:
      An instance of `tf.Session()`, as the session this iteration works on.

    iter_ops:
      List of `Op`s to be iterated. Ensure that it has been initialized.

    feed_dict:
      A `feed_dict` associated to the `tf.placeholder`s needed by the `iter_ops`.

    summarizer:
      A "summary op" that summarizes the graph, e.g. `tf.summary.merge_all`,
      optional.

    writer:
      An instance of `tf.summary.FileWriter` that writes the summary into disk.
      If the `summarizer` is `None` (as default), then this argument is useless,
      optional.

    global_step_val:
      `int` or `None`, as the value of global-step, optional.

    options:
      A `[RunOptions]` protocol buffer or `None`, as the associated argument of
      `tf.Session.run()`, optional.

    run_metadata:
      A `[RunMetadata]` protocol buffer or `None`, as the associated argument of
      `tf.Session.run()`, optional.

  Returns:
    List of the values of the `iter_ops`.
  """

  # Get `fetches`
  fetches = [op for op in iter_ops]
  if summarizer is not None:
    fetches.append(summarizer)

  # Iterate in one step and get values
  fetch_vals = sess.run(fetches,
                        feed_dict=feed_dict,
                        options=options,
                        run_metadata=run_metadata)

  # Write to TensorBoard
  if summarizer is not None and writer is not None:
    _, summary = fetch_vals
    writer.add_summary(summary, global_step=global_step_val)

  # Return the values of `iter_ops`
  if summarizer is not None:
    # The last element of `fetch_vals` will be the `summary`
    iter_op_vals = fetch_vals[:-1]
  else:
    iter_op_vals = fetch_vals

  return iter_op_vals



class BaseTrainer(object):
  """Abstract base class of trainer that supplements the `iterate`.

  In addition, the `save` and `restore`, as well as their essential instance of
  `tf.train.Saver`, will be defined, since initialization is entangled with them.

  Args:
    graph:
      An instance of `tf.Graph`, as the graph to be trained.

    sess:
      An instance of `tf.Session` or `None`, optional. If not `None`, the
      method `self.get_sess()` will not be called.

    sess_config:
      Optional. A ConfigProto protocol buffer with configuration options for
      the session.

    sess_target:
      Optional. The execution engine to connect to. Defaults to using an
      in-process engine. See Distributed TensorFlow for more examples.

    init_global_step:
      `int` as the initial value of global step, optional.

    initializer:
      A TenosrFlow initializer, or `None`, optional. If `None`, then using
      `tf.global_variables_initializer` as the employed initializer.

  Attributes:
    `graph`, `sess`, `saver`, `global_step`, `initializer`.

  Methods:
    create_saver:
      Abstract method. Returns an instance `tf.train.Saver()`.

    create_sess:
      Returns an instance of `tf.Session()` as `self.sess`.

    get_sess:
      Returns an instance of `tf.Session()` as the argument `sess` of
      `iterate()`.

    get_iter_ops:
      Abstract method. Returns list of ops as the argument `iter_ops` of
      `iterate()`.

    get_summarizer:
      Abstract method. Returns TensorFlow `Op` or `None` as the argument
      `summarizer` of `iterate()`.

    get_writer:
      Abstract method. Returns an instance of `tf.summary.FileWriter` or `None`
      as the argument `writer` of `iterate()`.

    save:
      Abstract method. Saves the checkpoint of `self.sess` to disk, or do nothing
      is none is to be saved.

    restore:
      Abstract method. Restores the checkpoint to `self.sess` from disk, or do
      nothing is none is to be saved.

    get_global_step_val:
      Returns an `int` as the temporal value of global step.

    iter_body:
      The body of iteration. It gets the arguments needed by `iterate()` and
      runs `iterate()` once. Also, it increments the value of `self.global_step`.

    train:
      As the trainer trains.
  """

  def __init__(self, graph, sess=None, sess_config=None,
        sess_target='', init_global_step=0, initializer=None):
    """This function shall be called after all codes in the `__init__()` of
    any class that inherits this abstract base class. For two reasons:

      1. Session shall be created after the graph has been completely built up
         (thus will not be modified anymore). The reason is that the partition
         of the resources and the edges of the graph in the session is optimized
         based on the graph. Thus the graph shall not be modified after the
         session having been created.

      2. Some operation, like initializer, shall be placed in the end of the
         graph.
    """

    # Do something that initializes a subclass that inherits this abstract base
    # class, and then call `super().__init__()` that runs the below.

    # Added name-scope "auxillary_ops" into `self.graph`.
    # Building of `iter_ops` may need `self.global_step`, which thus shall be
    # defined in front.
    with self.graph.as_default():
      with tf.name_scope('auxillary_ops'):
        with tf.name_scope('increase_global_step_op'):
          self.global_step = tf.Variable(
              init_global_step, trainable=False, name='global_step')
          self.increase_global_step_op = self.global_step.assign_add(1)

    # For saving and restoring, shall be located after introducing all variables
    # that are to be saved.
    self.saver = self.create_saver()

    # Initializer shall be placed in the end of the graph.
    # XXX: Why?
    with self.graph.as_default():
      with tf.name_scope('auxillary_ops'):
        with tf.name_scope('initializer'):
          self.initializer = tf.global_variables_initializer() \
                            if initializer is None else initializer

    if sess is None:
      self.sess_config = sess_config
      self.sess_target = sess_target
      self.sess = self.create_sess()
    else:
      self.sess = sess

    # Restore checkpoint to `self.sess` by `self.saver`
    self.restored = self.restore()


  @abc.abstractmethod
  def create_saver(self):
    """Abstract method. Returns an instance `tf.train.Saver()`.

    NOTE:
      Saver shall be initialized within `self.graph`.
    """
    pass


  @abc.abstractmethod
  def create_sess(self):
    """Abstract method. Returns an instance of `tf.Session()` as `self.sess`.
    This method will be called if and only if the argument `sess` in the
    `self.__init__()` is `None` (as default)."""
    pass


  def get_sess(self):
    """Returns an instance of `tf.Session()` as the argument `sess` of
    `iterate()`.

    Since there's only one session is needed through out the training process,
    we just return the `self.sess` created by method `self.create_sess`.
    """
    return self.sess


  @abc.abstractmethod
  def get_iter_ops(self):
    """Abstract method. Returns list of ops as the argument `iter_ops` of
    `iterate()`."""
    pass


  @abc.abstractmethod
  def get_summarizer(self):
    """Abstract method. Returns TensorFlow `Op` or `None` as the argument
    `summarizer` of `iterate()`."""
    pass


  @abc.abstractmethod
  def get_writer(self):
    """Abstract method. Returns an instance of `tf.summary.FileWriter` or `None`
    as the argument `writer` of `iterate()`."""
    pass


  @abc.abstractmethod
  def save(self):
    """Abstract method. Saves the checkpoint of `self.sess` to disk, or do
    nothing is none is to be saved."""
    pass


  @abc.abstractmethod
  def restore(self):
    """Abstract method. Restores the checkpoint to `self.sess` from disk, or do
    nothing is none is to be saved.

    Returns:
      `bool`, being `True` if sucessfully restored from checkpoint; else `False`.
    """
    pass


  def get_global_step_val(self):
    """Returns an `int` as the temporal value of global step."""
    global_step_val = tf.train.global_step(self.sess, self.global_step)
    return global_step_val


  def iter_body(self, feed_dict_generator, options, run_metadata):
    """The body of iteration. It gets the arguments needed by `iterate()` and
    runs `iterate()` once. Also, it increments the value of `self.global_step`.

    Appending anything into this `iter_body()` can be simply archived by re-
    implementing `iter_body()` with `super().iter_body(...)`.
    """

    # Get the arguments needed by `iterate()`
    sess = self.get_sess()
    iter_ops = self.get_iter_ops()
    feed_dict = next(feed_dict_generator)
    summarizer = self.get_summarizer()
    writer = self.get_writer()
    global_step_val = self.get_global_step_val()

    # Run `iterate()` once
    iterate(sess, iter_ops, feed_dict, summarizer=summarizer,
            writer=writer, global_step_val=global_step_val,
            options=options, run_metadata=run_metadata)

    # Also, increment the value of `self.global_step`
    self.sess.run(self.increase_global_step_op)


  def train(self, n_iters, feed_dict_generator,
            options=None, run_metadata=None, verbose=True):
    """As the trainer trains.

    Args:
      n_iters:
        `int`, as the number of iterations.

      feed_dict_generator:
        A generator that emits a feed_dict at each calling of `next()`.

      options:
        A `[RunOptions]` protocol buffer or `None`, as the associated argument
        of `tf.Session.run()`, optional.

      run_metadata:
        A `[RunMetadata]` protocol buffer or `None`, as the associated argument
        of `tf.Session.run()`, optional.

      verbose:
        `bool`.
    """

    try:
      if not self.restored:
        self.sess.run(self.initializer)
        if verbose:
          print('INFO - Restored.')
    except NameError:
      # meaning that `self.restored` is not defined, thus not restored.
      self.sess.run(self.initializer)
      if verbose:
        print('INFO - Initialize without restoring.')

    if verbose:
      global_step_val = tf.train.global_step(self.sess, self.global_step)
      print('INFO - Start training at global step {}.'.format(global_step_val))

    # Iterations
    for i in tqdm(range(n_iters)):  # XXX

      try:
        self.iter_body(feed_dict_generator, options, run_metadata)

      except StopIteration:
        # Meaning that the `feed_dict_generator` has been exhausted.
        print('INFO - No more training data to iterate.')
        break

    # Save the checkpoint to disk at the end of training
    self.save()
