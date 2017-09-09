#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
---------

This is an abstract class of TensorFlow trainer for general optimizational model,
e.g. neural network. It supports the most general training process, employing
arbitrary collection of data-sets (e.g. training data, validation data, testing
data, etc).

Explicitly, for any given optimizational model on TensorFlow, this trainer makes,
during the training:

    1. Train the model by calling its optimizer while feeding in training data.

    2. validation, testing, etc, are established at the same time.

    3. Save graph and checkpoint.
"""


import tensorflow as tf
import numpy as np
import os
from tools import ensure_directory




class Trainer:
    """
    Train the model by feeding training, vaidation, testing data, parameters, etc.

    Args:
        pnn:
            `PostNN` instance.

        params: dict
            Be of the form `{'<parameter>': <value>}`, where `<parameter>` shall
            be the same as that in `self.model.<parameter>`. This `<parameter>`
            is a `tf.constant` in the `self.model.graph`, which is to be tuned
            if essential.

            (Generally, we will collect all parameters, as `tf.constant`, into the
            "parameters" name-scope, thus being obvious in `tensorboard` "GRAPH".)

            (CAUTION, as a rule of Python, the `<parameter>` shall be quoted as a
            string.)

            Being {} if there is no parameter in `self.model` to be tuned.

        batch_generator: iter
            Calling `next(batch_generator)` yields a mini-batch of data for
            training.

        max_training_steps: int
            The training will start at a restored checkpoint, as its inital step,
            the practical maximum of training steps is the sum of the initial
            step and the `max_training_steps`.

        skip_step: int
            While training, the trainer will save the checkpoint for every
            `skip_step` steps.

        path_to_checkpoint: str
            Path to the `checkpoint` (*.ckpt) file which restore (if exists)
            or save checkpoint of training. It shall be of the form:

                '<path_to_checkpoints_dir>/<model_name>.ckpt'.

        path_to_graph: str
            Path to the `logdir` of `tensorboard`, in which the training summary
            of the model is to be saved.

    Methods:
        train():
            Do the training.
    """

    def __init__(self,
                 pnn,
                 params,
                 batch_generator,
                 max_training_steps,
                 skip_step,
                 path_to_graph,
                 path_to_checkpoint):

        self.pnn = pnn
        self.parameters = parameters
        self.batch_generators = batch_generators
        self.max_training_steps = max_training_steps
        self.skip_step = skip_step
        self.path_to_graph = path_to_graph
        self.path_to_checkpoint = path_to_checkpoint

        self._sess = tf.Session(graph=self.model.graph)


    def train(self):
        """ Train the model. """

        with self._sess:

            self._prepare_for_training()

            training_steps = range(
                    self._initial_step,
                    self._initial_step + self.max_training_steps
                    )
            for step in training_steps:

                self._train_by_feeding(step)

                if (step + 1) % self.skip_step == 0:

                    self._save_checkpoint(step)

                else:
                    pass

            self._postpare_for_training()


    def _prepare_for_training(self):
        """
        General setup of preparing for a training of TensorFlow model.

        Explicitly:
            1. create writers for each data-set (training, validation, testing,
               etc);

            2. initialize global step, which keep track of checkpoint;

            3. create saver;

            4. initialize all `tf.Variable`s in the model.

            5. get the latest checkpoint. if exists, then continue the training
               from the latest checkpoint.
        """

        # Create writer for each data-set
        # (i.e. training, validation, and testing, etc).
        self._writers = []
        for i, _ in enumerate(self.batch_generators):
            # Summery of different data shall be written into
            # different directory, helpful for `tensorboard`.
            writer = tf.summary.FileWriter(
                         os.path.join(self.path_to_graph,
                                      'dataset_{0}/'.format(i)
                                      ),
                         self._sess.graph
                         )
            self._writers.append(writer)

        # global_step to keep track of checkpoint
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

        # Create saver
        self._saver = tf.train.Saver()

        # Initialize all `tf.Variable`s in one go
        self._sess.run(tf.global_variables_initializer())

        # Get checkpoint
        # CAUTION that the arg of `get_checkpoint_state` is `checkpoint_dir`,
        # i.e. the directory of the `checkpoint` to be restored from.
        ckpt = tf.train.get_checkpoint_state(
                os.path.dirname(self.path_to_checkpoint)
                )
        self._initial_step = 0

        # If that checkpoint exists, then restore from the checkpoint
        if ckpt and ckpt.model_checkpoint_path:

            self._saver.restore(self._sess,ckpt.model_checkpoint_path)

            # A rude way of reading the step of the latest checkpoint.
            # And assign it as the initial step of the later training.
            self._initial_step = \
                int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

        else:
            pass


    def _train_by_feeding(self, global_step):
        """
        Train the model by feeding the data (`self.model.inputs`,
        `self.model.targets`, and `self.model.<parameter>`s if
        essential) from `self.batch_generators` and `self.parameters`.

        Args:
            global_step: int
        """

        for i, batch_generator in enumerate(self.batch_generators):

            # Generate feed_dict
            feed_dict = {}

            for param in self.parameters:
                feed_dict[getattr(self.model, param)] = self.parameters[param]

            inputs, targets = batch_generator.next()
            feed_dict[self.model.inputs] = inputs 
            feed_dict[self.model.targets] = targets

            # Run the optimizer while feeding
            if i == 0:  
                # Meaning that it's training data. For training data, we shall
                # update the parameters (i.e. `trainable` `tf.Variable`s)in
                # self.model.
                # (Recall that we have demanded to place the batch-generator of
                #  training data to the first place of the list of generators.)
                self._sess.run(self.model.optimizer,
                               feed_dict=feed_dict
                               )
            else:
                pass

            # Write to `tensorboard`
            summary = self._sess.run(self.model.summary,
                                     feed_dict=feed_dict
                                     )
            self._writers[i].add_summary(summary,
                                         global_step=global_step
                                         )


    def _save_checkpoint(self, global_step):
        """
        Args:
            global_step: int
        """

        # `tf.saver` cannot automatically mkdir, so
        ensure_directory(os.path.dirname(self.path_to_checkpoint))

        self._saver.save(self._sess,
                         self.path_to_checkpoint,
                         global_step=global_step
                         )


    def _postpare_for_training(self):
        """
        General setup of postparing for a training of TensorFlow model.

        Explicitly:
            1. write the training summaries to disk (for `tensorboard`);

            2. close writers

            3. close session.
        """

        # While ending:
        for writer in self._writers:

            # Write the summaries to disk
            writer.flush()

            # Close the SummaryWriter
            writer.close()

        # Close the session
        self._sess.close()
