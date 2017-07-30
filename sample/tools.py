#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import time



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
