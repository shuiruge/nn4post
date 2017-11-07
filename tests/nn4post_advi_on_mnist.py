#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Shallow neural network by `Edward ` on MNIST dataset.
"""


import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import pickle

import sys
sys.path.append('../sample/')
from tools import Timer
import mnist
from nn4post_advi import build_inference



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # turn off the TF noise.
ed.set_seed(42)  # for debugging.



# DATA
NOISE_STD = 0.1
BATCH_SIZE = 16
mnist_ = mnist.MNIST(NOISE_STD, BATCH_SIZE)
batch_generator = mnist_.batch_generator()


graph = tf.get_default_graph()


N_C = 1
graph, ops = build_inference(N_C, base_graph=graph)
