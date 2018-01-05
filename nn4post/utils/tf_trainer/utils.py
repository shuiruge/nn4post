#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Description
---------
Helper functions for TensorFlow.

Focked from _TensorFlow for Machine Intelligence_, chapter
_8. Helper Functions, Code Structure, and Classes_.

`define_scope()` and its helper `doublewrap()` are focked from
[here](https://danijar.github.io/structuring-your-tensorflow-models).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools
import errno
import os
import tensorflow as tf



def ensure_directory(directory):
    """
    Check whether the `directory` exists or not. If not, then create
    this directory.

    Args:
        directory: str
    Returns:
        None
    """

    directory = os.path.expanduser(directory)

    try:
        os.makedirs(directory)

    except OSError as e:

        if e.errno != errno.EEXIST:

            raise e
    
    return None
