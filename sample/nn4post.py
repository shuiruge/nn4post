#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------

"""


import tensorflow as tf
# -- `contrib` module in TensorFlow version: 1.2
from tensorflow.contrib.distributions import \
    Categorical, Mixture, MultivariateNormalDiag
from tensorflow.contrib.distributions import softplus_inverse
from tensorflow.contrib.bayesflow import entropy




# --- Main Class ---

class PostNN(object):
    """ Main class of "neural network for posterior" ("nn4post" for short).
    
    Args:
        num_peaks: int
        
        dim: int
        
        num_samples: int
        
    Attributes:
        graph: tf.Graph()
        
    Method:
        compile
        fit
        
    TODO: complete it.
    TODO: write `self.fit()`.
    TODO: write `self.inference()`.
    """
    
    def __init__(self, num_peaks, dim, num_samples=1000):
        
        self._num_peaks = num_peaks
        self._dim = dim
        self._num_samples = num_samples
        
        self.graph = tf.Graph()
        
        # -- Parameters
        self._float = tf.float32
        
    
    @staticmethod
    def _chi_square(model, data_x, data_y, data_y_error, params):
        """ Denote :math:`f` as the `model`, :math:`\theta` as the `params`,
            and :math:`\sigma` as the `data_y_error`, we have
            
        ```math
        
        \chi^2 \left( (x, y, \sigma), \theta \right) = -\frac{1}{2}
            \sum_i^N \left( \frac{y_i - f(x_i, \theta)}{\sigma_i} \right)^2
        ```
        
        TODO:
            Self-adaptively add some constant (like `prob(data)`) so that the
            uppder of `_chi_square()` can be bounded, as the number of data
            increases, for avoiding overflow of `_chi_square()`, thus the
            `elbo` in blow.
                        
        Args:
            model:
                Callable, mapping from `(x, theta)` to `y`; wherein `x` is a
                `Tensor` representing the input data to the `model`, having
                shape `[batch_size, ...]` (`...` is for any sequence of `int`)
                and any dtype, so is the `y`; however, the `theta` must have
                the shape `[self.get_dim()]` and dtype `self._float`.
                
            data_x: `Tensor` described as above.
            
            data_y: `Tensor` described as above.
            
            data_y_error: `Tensor` as `data_y`.
            
            params:
                `Tensor` with shape `[self.get_dim()]` and dtype `self._float`.
            
        Returns:
            `Tensor` with shape `[]` and dtype `self._float`.
        """
        
        noise = tf.subtract(data_y, model(data_x, params))
        
        return tf.reduce_sum(-0.5 * tf.square( noise / data_y_error ))
    
    
    def compile(self,
                model,
                log_prior=lambda theta: 0.0,
                optimizer=tf.train.RMSPropOptimizer,
                learning_rate=0.01,
                ):
        """ Set up the (TensorFlow) computational graph.
        
        CAUTION ERROR:
            Using `GradientDescentOptimizer` will increase the loss and
            raise an ERROR after several training-steps. However, e.g.
            `AdamOptimizer` and `RMSPropOptimizer` naturally saves this.
        
        Args:
            model:
                Callable, mapping from `(x, theta)` to `y`; wherein `x` is a
                `Tensor` representing the input data to the `model`, having
                shape `[batch_size, ...]` (`...` is for any sequence of `int`)
                and any dtype, so is the `y`; however, the `theta` must have
                the shape `[self.get_dim()]` and dtype `self._float`.
                
            log_prior:
                Callable, mapping from `theta` to `self._float`, wherein
                `theta` is a `Tensor` which must have the shape shape
                `[self.get_dim()]` and dtype `self._float`.
                
            optimizer:
                Optimizer object of module `tf.train`, c.f. the "CAUTION ERROR".
                
            learning_rate:
                `float`, the learning rate of the `optimizer`.

        NOTE:
            To generate the `model`, you shall first write the model in
            TensorFlow without caring about its `params` argument. Then you can
            write a `parse_params()` helper function, which parse a 1-D `Tensor`
            to a list of `Tensor`s with the correspoinding shapes. This can be
            established directly by `tf.split()` and `tf.reshape()`, as long as
            you have patiently find out the correct shapes for each `Tensor` in
            the parsed list.
        """
        
        with self.graph.as_default():
            
            with tf.name_scope('data'):
            
                # shape: [batch_size, *model_input]
                self.x = tf.placeholder(dtype=self._float,
                                        name='x')
                # shape: [batch_size, *model_output]
                self.y = tf.placeholder(dtype=self._float,
                                        name='y')
                # shape: [batch_size, *model_output]
                self.y_error = tf.placeholder(dtype=self._float,
                                              name='y_error')
            
            with tf.name_scope('log_p'):
                
                def chi_square_on_data(theta):
                    """ Chi-square based on the data from 'data_source'.
                    
                    Args:
                        theta: `Tensor` with shape `[self.get_dim()]` and dtype
                               `self._float`.
                    Returns:
                        `self._float`.
                    """
                    return self._chi_square(
                        model=model, data_x=self.x, data_y=self.y,
                        data_y_error=self.y_error, params=theta)
                
                def log_posterior(theta):
                    """ ```math
                        \ln\textrm{posterior} = \chi^2 + \ln\textrm{prior}
                        ```
                    """
                    return chi_square_on_data(theta) + log_prior(theta)
                                   
                def log_p(thetas):
                    """ Vectorized `log_posterior()`.
                    
                    Args:
                        thetas: `Tensor` with shape `[None, self.get_dim()]`
                                and dtype `self._float`.
                        
                    Returns:
                        `Tensor` with shape `[None]` and dtype `self._float`.
                        The two `None` shall both be the same value (e.g. both
                        being `self.get_num_samples()`).
                    """
                    return tf.map_fn(log_posterior, thetas,
                                     name='log_p_as_vectorized')
            
            with tf.name_scope('trainable_variables'):
                                    
                a_shape = [self.get_num_peaks()]
                self.a = tf.Variable(
                    initial_value=tf.ones(a_shape),
                    dtype=self._float,
                    name='a')
                
                mu_shape = [self.get_num_peaks(), self.get_dim()]
                self.mu = tf.Variable(
                    initial_value=tf.zeros(mu_shape),
                    dtype=self._float,
                    name='mu')
                
                zeta_shape = [self.get_num_peaks(), self.get_dim()]
                self.zeta = tf.Variable(
                    initial_value=softplus_inverse(tf.ones(zeta_shape)),
                    dtype=self._float,
                    name='zeta')
            
            with tf.name_scope('CGMD_model'):
                
                with tf.name_scope('model_parameters'):
                    
                    self.weight = tf.nn.softmax(self.a,
                                                name='weight')
                    self.sigma = tf.nn.softplus(self.zeta,
                                                name='sigma')
                
                with tf.name_scope('categorical'):
                    
                    cat = Categorical(probs=self.weight,
                                      name='cat')
                
                with tf.name_scope('Gaussian'):
                    
                    mu_list = tf.unstack(self.mu, name='mu_list')
                    sigma_list = tf.unstack(self.sigma, name='sigma_list')
                    components = [
                        MultivariateNormalDiag(
                            loc=mu_list[i],
                            scale_diag=sigma_list[i],
                            name='Gaussian_{0}'.format(i))
                        for i in range(self.get_num_peaks())]
                
                with tf.name_scope('mixture'):
                    
                    self.cgmd = Mixture(cat=cat,
                                        components=components,
                                        name='CGMD')
                            
            with tf.name_scope('loss'):
                
                theta_samples = self.cgmd.sample(self.get_num_samples(),
                                                 name='theta_samples')
                elbo = entropy.elbo_ratio(log_p, self.cgmd, z=theta_samples,
                                          name='ELBO')
                
                # In TensorFlow, ELBO is defined as E_q [log(p / q)], rather
                # than as E_q [log(q / p)] as WikiPedia. So, the loss would be
                # `-1 * elbo`
                self.loss = tf.multiply(-1.0, elbo,
                                        name='loss')
        
            with tf.name_scope('optimize'):
                
                self.optimize = optimizer(learning_rate).minimize(self.loss)
            
            with tf.name_scope('summary'):
                
                tf.summary.scalar('loss', self.loss)
                tf.summary.histogram('histogram_loss', self.loss)
                self.summary = tf.summary.merge_all()
                   
        print('INFO - Model compiled.')
                
    
    def fit(self,
            batch_generator,
            epochs,
            logdir=None,
            verbose=False,
            skip_steps=100,
            debug=False):
        """
        TODO: complete docstring.  
        TODO: add `tf.train.Saver()`, `global_step`, etc.
        """
        
        sess = tf.Session(graph=self.graph)
        
        if debug:
            from tensorflow.python import debug as tf_debug
            sess = tf_debug.LocalCLIDebugWrapperSession(
                sess, thread_name_filter='MainThread$')
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
                    
        if logdir != None:
            self._writer = tf.summary.FileWriter(logdir, self.graph)
        
        with sess:
            
            sess.run(tf.global_variables_initializer())
            
            for step in range(epochs):
                
                x, y, y_error = next(batch_generator)
                feed_dict = {self.x: x, self.y: y, self.y_error: y_error}
                
                if logdir == None:
                    _, loss_val = sess.run(
                            [self.optimize, self.loss],
                            feed_dict=feed_dict)
                    
                else:
                    _, loss_val, summary_val = sess.run(
                            [self.optimize, self.loss, self.summary],
                            feed_dict=feed_dict)
                    self._writer.add_summary(summary_val, global_step=step)
                                
                if verbose:
                    if (step+1) % skip_steps == 0:
                        print('step: {0}'.format(step+1))
                        print('loss: {0}'.format(loss_val))
                        print('-----------------------\n')
                        
            self._a_val, self._mu_val, self._zeta_val = \
                sess.run([self.a, self.mu, self.zeta])
            
    
    def inference(self, num_samples):
        """
        TODO: complete docstring.  
        """
        
        sess = tf.Session(graph=self.graph)
        
        with sess:
            
            theta_vals = sess.run(tf.unstack(self.cgmd.sample(num_samples)))
            
        return theta_vals
    
    
    
    # -- Get-Functions
                
    def get_num_peaks(self):
        return self._num_peaks
    
    def get_dim(self):
        return self._dim
    
    def get_num_samples(self):
        return self._num_samples
    
    def get_variables(self):
        variables = [self.a, self.mu, self.zeta]
        sess = tf.Session(graph=self.graph)
        with sess:
            variable_values = sess.run(variables)
        return variable_values