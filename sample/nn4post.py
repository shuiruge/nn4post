#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Implementation of the model illustrated in '../docs/nn4post.pdf', via
TensorFlow.

Documentation
-------------
C.f. '../docs/nn4post.pdf'.
"""


import tensorflow as tf
import numpy as np
# -- `contrib` module in TensorFlow version: 1.2
from tensorflow.contrib.distributions import \
    Categorical, Mixture, MultivariateNormalDiag
from tensorflow.contrib.bayesflow import entropy




# --- Main Class ---

class PostNN(object):
    """ Main class of "neural network for posterior" ("nn4post" for short).
    
    DOCUMENTATION:
        'nn4post/docs/nn4post.pdf'.
    
    Args:
        num_peaks: int
            Number of Gaussian peaks, that is, the number of categories in the
            categorical distribution (i.e. the :math:`N_c` in documentation).
        
        dim: int
            The dimension of parameter-space. (I.e. the :math:`d` in
            documentation.)
                
        model:
            Callable, mapping from `(x, theta)` to `y`; wherein `x` is a
            `Tensor` representing the input data to the `model`, having
            shape `[batch_size, ...]` (`...` is for any sequence of `int`)
            and any dtype, so is the `y`; however, the `theta` must have
            the shape `[self.get_dim()]` and dtype `self._float`. (I.e. the
            :math:`f` in documentation.)
            
            To generate the `model`, you shall first write the model in
            TensorFlow without caring about its `params` argument. Then you can
            write a `parse_params()` helper function, which parse a 1-D `Tensor`
            to a list of `Tensor`s with the correspoinding shapes. This can be
            established directly by `tf.split()` and `tf.reshape()`, as long as
            you have patiently find out the correct shapes for each `Tensor` in
            the parsed list.
            
        log_prior:
            Callable, mapping from `theta` to `self._float`, wherein
            `theta` is a `Tensor` which must have the shape shape
            `[self.get_dim()]` and dtype `self._float`. (I.e. the
            :math:`\ln p(\theta)` in documentation.)
            
        num_samples: int
        
        debug: bool
            If `True`, then employ the `tfdbg`.
        
    Attributes:
        graph: tf.Graph()
            Computational graph of `PostNN()`. Compiled after calling
            `self.compile()`.
            
        `Tensor`s within `self.graph`.

        
    Method:
        set_vars
        compile
        fit
        
    TODO: complete it.
    TODO: write `self.fit()`.
    TODO: write `self.inference()`.
    """
    
    def __init__(self,
                 num_peaks,
                 dim,
                 model,
                 log_prior=lambda x: 0.0,
                 num_samples=1000,
                 debug=False):
        
        self._num_peaks = num_peaks
        self._dim = dim
        self._model = model
        self._log_prior = log_prior
        self._num_samples = num_samples
        self._debug = debug
        
        self.graph = tf.Graph()
        
        
        # --- Parameters ---
        
        self._float = tf.float32
        
        self._a_shape = [self.get_num_peaks()]
        self._mu_shape = [self.get_num_peaks(), self.get_dim()]
        self._zeta_shape = [self.get_num_peaks(), self.get_dim()]
        
        
        # -- initialize the values of variables of CGMD.
        self._a_val = np.ones(self._a_shape)
        self._mu_val = np.random.normal(scale=1.0,
                                        size=self._mu_shape)
        # To make `softplus(self._init_zeta) == np.ones(self._zeta_shape)`
        self._zeta_val = np.log((np.e-1) * np.ones(self._zeta_shape))
        
        
    def _check_var_shape(self, a, mu, zeta):
        """ Check the shape of varialbes (i.e. `a`, `mu`, and `zeta`).
        
        Args:
            a, mu, zeta: array-like.
        """
        
        # -- Check Shape
        shape_error_msg = 'ERROR: {0} expects the shape {2}, but given {1}.' 
        a_shape, mu_shape, zeta_shape = self.get_var_shapes()
        assert a.shape == a_shape, \
            shape_error_msg.format('a', a.shape, a_shape)
        assert mu.shape == mu_shape, \
            shape_error_msg.format('mu', mu.shape, mu_shape)
        assert zeta.shape == zeta_shape, \
            shape_error_msg.format('zeta', zeta.shape, zeta_shape)
        
        # -- Check Dtype
        dtype_error_msg = 'ERROR: {0} expects the dtype {2}, but given {1}.'
        var_dtype = self.get_var_dtype()
        assert a.dtype == self._float, \
            dtype_error_msg.format('a', a.dtype, var_dtype)
        assert mu.dtype == self._float, \
            dtype_error_msg.format('mu', mu.dtype, var_dtype)
        assert zeta.dtype == self._float, \
            dtype_error_msg.format('zeta', zeta.dtype, var_dtype)
        

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
        
        return tf.reduce_sum( -0.5 * tf.square(noise/data_y_error) )
    
    
    def _create_session(self):
        """ Create a `tf.Session()` object that runs the `self.graph`.
        
        NOTE: can only be called after `self.compile()`.
        
        Returns:
            `tf.Session()` object that runs the `self.graph`.
        """
        
        sess = tf.Session(graph=self.graph)
        
        if self._debug:
            from tensorflow.python import debug as tf_debug
            sess = tf_debug.LocalCLIDebugWrapperSession(
                sess, thread_name_filter='MainThread$')
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            
        return sess
    
    
    def compile(self,
                optimizer=tf.train.RMSPropOptimizer,
                learning_rate=0.01,
                init_vars=None,
                ):
        """ Set up the (TensorFlow) computational graph.
        
        CAUTION ERROR:
            Using `GradientDescentOptimizer` will increase the loss and
            raise an ERROR after several training-steps. However, e.g.
            `AdamOptimizer` and `RMSPropOptimizer` naturally saves this.
        
        Args:                
            optimizer:
                Optimizer object of module `tf.train`, c.f. the "CAUTION ERROR".
                
            learning_rate:
                `float`, the learning rate of the `optimizer`.
                
            init_vars:
                "Initial value of variables (i.e. `a`, `mu`, and `zeta`)", as
                a tuple of numpy array or `None`. If `None`, use default values.
                If not `None`, then they shall be the numpy arries with the
                shapes of `self.get_a_shape()`, `self.get_mu_shape()`,
                and `self.get_zeta_shape()` respectively, and dtypes of
                `self.get_var_dtype()` uniformly.
                
        Modifies:
            `self.graph` and `self.*` therein; `self._sess`
        """
        
        # --- Construct TensorFlow Graph ---
        with self.graph.as_default():
                                    
            with tf.name_scope('data'):
                
                var_dtype = self.get_var_dtype()
                
                # shape: [batch_size, *model_input]
                self.x = tf.placeholder(dtype=var_dtype,
                                        name='x')
                # shape: [batch_size, *model_output]
                self.y = tf.placeholder(dtype=var_dtype,
                                        name='y')
                # shape: [batch_size, *model_output]
                self.y_error = tf.placeholder(dtype=var_dtype,
                                              name='y_error')
                
            
            with tf.name_scope('log_p'):
                
                def chi_square_on_data(theta):
                    """ Chi-square based on the data from 'data_source'.
                    
                    Args:
                        theta: `Tensor` with shape `[self.get_dim()]` and dtype
                               `self._float`.
                               
                    Returns:
                        `Tensor` with shape `[]` and dtype `self._float`.
                    """
                    model = self.get_model()
                    return self._chi_square(
                        model=model, data_x=self.x, data_y=self.y,
                        data_y_error=self.y_error, params=theta)
                    
                
                def log_posterior(theta):
                    """ ```math
                    
                        \ln \textrm{posterior} = \chi^2 + \ln \textrm{prior}
                        ```
                        
                    I.e. the :math:`\ln p(\theta \| D)` in documentation.
                    """
                    log_prior = self.get_log_prior()
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
                
                init_a, init_mu, init_zeta = self.get_vars()
                var_dtype = self.get_var_dtype()
                                                    
                self.a = tf.Variable(
                    initial_value=init_a,
                    dtype=var_dtype,
                    name='a')
                
                self.mu = tf.Variable(
                    initial_value=init_mu,
                    dtype=var_dtype,
                    name='mu')
                
                self.zeta = tf.Variable(
                    initial_value=init_zeta,
                    dtype=var_dtype,
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
                
                
            with tf.name_scope('model_output'):
                
                self.model_output = tf.reduce_mean(
                    tf.map_fn(lambda theta: self._model(self.x, theta),
                              theta_samples),
                    axis=0,
                    name='model_output')
                
            
            with tf.name_scope('summary'):
                
                tf.summary.scalar('loss', self.loss)
                tf.summary.histogram('histogram_loss', self.loss)
                self.summary = tf.summary.merge_all()
                
                
            with tf.name_scope('other_ops'):
                
                self.init = tf.global_variables_initializer()
                
        
        print('INFO - Model compiled.')   
             
    
    def fit(self,
            batch_generator,
            epochs,
            logdir=None,
            verbose=False,
            skip_steps=100):
        """
        TODO: complete docstring.  
        TODO: add `tf.train.Saver()`, `global_step`, etc.
        """
        
        if logdir is not None:
            self._writer = tf.summary.FileWriter(logdir, self.graph)
        
        sess = self.get_session()

        with sess.as_default():
            
            sess.run(self.init)
            
            for step in range(epochs):
                
                x, y, y_error = next(batch_generator)
                feed_dict = {self.x: x, self.y: y, self.y_error: y_error}
                
                if logdir is None:
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
            
            # Update the values of varialbes of CGMD
            self._a_val, self._mu_val, self._zeta_val = \
                sess.run([self.a, self.mu, self.zeta])
            
            # For convienence
            self._weight_val, self._sigma_val = \
                sess.run([self.weight, self.sigma])
    
    
    def predict(self, x):
        """
        TODO: complete this.  
        """
        
        sess = self.get_session()
        
        with sess.as_default():
        
            output_val = sess.run(self.model_output,
                                  feed_dict={self.x: x})
        
        return output_val
    
    
    def finalize(self):
        """ Release the deployed resource by TensorFlow. """
        
        try:
            # Write the summaries to disk
            self._writer.flush()
            # Close the SummaryWriter
            self._writer.close()
        except:
            print('INFO - No `SummaryWriter` to close.')
        # Close the Session
        self._sess.close()

    
    
    # -- Get-Functions
                
    def get_num_peaks(self):
        return self._num_peaks
    
    def get_dim(self):
        return self._dim
    
    def get_num_samples(self):
        return self._num_samples
        
    def get_var_shapes(self):
        """ Get the tensor-shape of the variables `a`, `mu`, and `zeta`.
        
        Returns:
            Tuple of lists, wherein each list represents a tensor-shape.
        """
        return (self._a_shape, self._mu_shape, self._zeta_shape)
    
    def get_var_dtype(self):
        """ Get the dtype of the variables `a`, `mu`, and `zeta`. All of them
            share the same dtype as convention.
            
        Returns:
            `tf.Dtype` object.
        """
        return self._float
    
    def get_model(self):
        """ Get the model (whose posteior is to be fitted by CGMD).
        
        Returns:
            Callable, as the model is.
        """
        return self._model
    
    def get_log_prior(self):
        """ Get the log_prior.
        
        Returns:
            Callable, as the log_prior is.
        """
        return self._log_prior
    
    def get_vars(self):
        """ Get tuple of numerical values (as numpy arraies) of variables,
            including `a`, `mu`, and `zeta`.
        
        CAUTION:
            Can only be called after an `self.fit()`.
            
        Returns:
            Tuple of numpy arraies, being the numerical values of `a`, `mu`,
            and `zeta`.
        """
        return (self._a_val, self._mu_val, self._zeta_val)
    
    def get_cgmd_params(self):
        """ Get tuple of numerical values (as numpy arraries) of CGMD
        parameters, including `weight`, `mu`, and `sigma`.
        
        CAUTION:
            Can only be called after an `self.fit()`.
            
        Returns:
            Tuple of numpy arraies, being the numerical values of `weight`,
            `mu`, and `sigma`.
        """
        return (self._weight_val, self._mu_val, self._sigma_val)
    
    def get_session(self):
        """ Get the `tf.Session()` that runs `self.graph`. No session has been
            created yet, then create and return one.
        
        Returns:
            `tf.Session()` object.
        """
        try:
            return self._sess
        except:
            print('INFO - created a `tf.Session()` object.')
            self._sess = self._create_session()
            return self._sess
        
    
    # -- Set-Functions
    def set_vars(self, vars_val):
        """ Set the values of variables of CMBD. This setting or re-setting can
            not work without re-calling the `self.compile()`.

        Args:
            vars_val:
                list of numpy array or `None`, as the values of the variables
                of CGMD.
                
        Modifies:
            `self._a_val`, `self._mu_val`, and `self._zeta_val`.
        """
        
        self._check_var_shape(*vars_val)
        
        # If passed this check without raising
        self._a_val, self._mu_val, self._zeta_val = vars_val