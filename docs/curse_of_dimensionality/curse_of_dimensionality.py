#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
A Monte Carlo implementation of the section "The Curse of Dimensionality" in
the documentation "/docs/nn4post.tm".
"""

import numpy as np
import matplotlib.pyplot as plt

# For debuging
SEED = 42
np.random.seed(SEED)

# Set the number of samples in the Monte Carlo simulation
N_SAMPLES = 10**2


def distance(x, y):
    """Euclidean distance between vector `x` and vector `y`."""
    return np.sqrt(np.sum(np.square( x - y )))


def get_y1(dim):
    """Get the point `y1`."""
    return np.ones([dim]) * (-1.0)


def get_y2(dim):
    """Get the point `y2`."""
    return np.ones([dim]) * 3.0


def get_samples(r, dim, n_samples):
    """Get the Monte Carlo samples.
    
    Args:
        r:
            Callable, mapping from `int` (as the dimension) to `float`, as the
            range of sampling.
            
        dim:
            `int`, as the number of dimension.
            
        n_samples:
            `int`, as the number of samples in the Monte Carlo simulation.
    
    Returns:
        List (with `n_samples` elements) of numpy array of the shape `[dim]`.
    """
    return [np.random.uniform(-1, 1, size=[dim]) * r(dim)
            for i in range(n_samples)]


def get_ratio_close_to_y1(r, dims, n_samples=N_SAMPLES):
    """Get the ratio of samples that sit closer to `y1` that  to `y2`.
    
    Args:
        r:
            Callable, mapping from `int` (as the dimension) to `float`, as the
            range of sampling.
            
        dims:
            List of `int`s, as the range of dimension.
            
        n_samples:
            `int`, as the number of samples in the Monte Carlo simulation.
            
    Returns:
        List of `float`s as the ratios.
    """
    
    ratio_close_to_y1 = []
    for dim in dims:
        y1 = get_y1(dim)
        y2 = get_y2(dim)
        samples = get_samples(r, dim, n_samples)
        
        n_close_to_y1 = 0
        for s in samples:
            if distance(s, y1) < distance(s, y2):
                n_close_to_y1 += 1
        ratio_close_to_y1.append(n_close_to_y1 / n_samples)
    
    return ratio_close_to_y1

# --- LET'S PLAY ---

# Set the range of sampling
r = lambda dim: np.sqrt(dim) * 5
# Set the range of dimension.
exps = np.arange(0, 16, 0.2)
dims = [int(np.exp(_)) for _ in exps]
ratio_close_to_y1 = get_ratio_close_to_y1(r, dims)

# Plot it out
plt.plot(exps, ratio_close_to_y1)
plt.xlabel('$\ln$(dimension)')
plt.ylabel('ratio close to $y_0$')
plt.show()
