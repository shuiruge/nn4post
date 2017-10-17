#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from time import time
import numpy as np
import tensorflow as tf



class Timer(object):
    """
    C.f. [here](https://www.huyng.com/posts/python-performance-analysis)
    """

    def __init__(self, verbose=True):
        self.verbose = verbose

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('=> elapsed time: %f secs' % self.secs)



def ensure_directory(directory):
    """
    Check whether the `directory` exists or not. If not, then create
    this directory.

    Args:
        directory: str
    """

    try:
        directory = os.path.expanduser(directory)
        os.makedirs(directory)

    except:
        pass



def get_accuracy(y_pred, y_true):
    """
    Args:
        y_pred:
            Numpy array with shape `[data_size]`.
        y_true:
            Numpy array with shape `[data_size]`.

    Returns:
        `float` as the accuracy.
    """
    corrects = [1 if int(p) == int(t) else 0
                 for p, t in list(zip(y_pred, y_true))]
    accuracy = np.mean(corrects)
    return accuracy



def get_variable_value_dict(sess):
    """
    Get all trainable variables and their current values in session `sess`.

    Args:
        sess:
            `tf.Session` object.

    Returns:
        `dict` object, with keys the variables' names (structured by TensorFlow),
        and values the associated variables' values in session `sess`.
    """

    variables = tf.trainable_variables()  # XXX: without figuring out graph?

    variable_names = [v.name for v in variables]
    values = sess.run(variables)
    name_to_val_list = list(zip(variable_names, values))
    return {name: val for name, val in name_to_val_list}




class TimeLiner(object):
    """ Focked from [here](https://github.com/ikhlestov/tensorflow_profiling/\
    blob/master/03_merged_timeline_example.py). """

    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)
