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


import os
import tensorflow as tf
import numpy as np
# -- `contrib` module in TensorFlow version: 1.3
from tensorflow.contrib.distributions import \
    Categorical, Mixture, MultivariateNormalDiagWithSoftplusScale
from tensorflow.contrib.bayesflow import entropy
from tools import ensure_directory


# -- For testing (and debugging)
tf.set_random_seed(42)
np.random.seed(42)



# --- Main Class ---

class Nn4post(object):
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

            Default is uniform prior.

        debug: bool
            If `True`, then employ the `tfdbg`.

        float_: str
            The dtype under running in `str`, e.g. `float32`.

    Attributes:
        graph: tf.Graph()
            Computational graph of `PostNN()`. Compiled after calling
            `self.compile()`.

        `Tensor`s within `self.graph`.

    Method:
        set_vars
        compile
        fit

    Remarks:

        - Set initial value of variable `a` by `random.normal()` gains much
          better result than by `ones()`.
        - Set initial value of variable `mu` by a larger `scale` makes things
          worse.
        - Set loss by `-1.0 * renyi_ratio` makes great instability while
          training.


    TODO:
        Self-adaptively add some constant (like `prob(data)`) so that the uppder
        of `_chi_square()` can be bounded, as the number of data increases, for
        avoiding overflow of `_chi_square()`, thus the `elbo` in blow.

    TODO:
        Is there some method that can estimate the :math:`p(D)`?


    TODO: complete it.
    """

    def __init__(self,
                 num_peaks,
                 dim,
                 model,
                 log_prior,
                 debug=False,
                 float_='float32',
                 dir_to_ckpt=None):

        self._num_peaks = num_peaks
        self._dim = dim
        self._model = model
        self._log_prior = log_prior
        self._debug = debug
        self._dir_to_ckpt = dir_to_ckpt
        if self._dir_to_ckpt is not None:
            ensure_directory(self._dir_to_ckpt)

        self.graph = tf.Graph()


        # -- Dtype
        if float_ == 'float32':
            self._float = tf.float32
            np_float = np.float32
        elif float_ == 'float64':
            self._float = tf.float64
            np_float = np.float64


        # -- Parameters
        self._a_shape = [self._num_peaks]
        self._mu_shape = [self._num_peaks, self._dim]
        self._zeta_shape = [self._num_peaks, self._dim]


        # -- initialize the values of variables of `cat_gauss_mix_dist`.
        self._a_val = np.zeros(shape=self._a_shape)
        self._a_val = self._a_val.astype(np_float)

        self._mu_val = np.random.normal(size=self._mu_shape)
        self._mu_val = self._mu_val.astype(np_float)

        # To make `softplus(self._init_zeta) == np.ones(self._zeta_shape)`
        self._zeta_val = np.log((np.e-1) * np.ones(self._zeta_shape))
        self._zeta_val = self._zeta_val.astype(np_float)


    def compile(self,
                optimizer=tf.train.RMSPropOptimizer,
                init_vars=None,
                ):
        """ Set up the (TensorFlow) computational graph.

        CAUTION ERROR:
            Using `GradientDescentOptimizer` will increase the loss and
            raise an ERROR after several training-steps. However, e.g.
            `AdamOptimizer` and `RMSPropOptimizer` naturally saves this.

        Args:
            optimizer:
                `tf.Optimizer` object of module `tf.train`, with the
                arguments fulfilled except for `learning_rate`.

                C.f. the "CAUTION ERROR".

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


            with tf.name_scope('parameters'):

                # shape: []
                self.batch_ratio = tf.placeholder(shape=[],
                                                  dtype=var_dtype,
                                                  name='batch_ratio')
                # shape: []
                self.learning_rate = tf.placeholder(shape=[],
                                                    dtype=self._float,
                                                    name='learning_rate')

                self.num_samples = tf.placeholder(shape=[],
                                                  dtype=tf.int32,
                                                  name='num_samples')


            with tf.name_scope('log_p'):

                def chi_square_on_data(theta):
                    """ Chi-square based on the data from 'data_source'.

                    CAUTION:
                        Have to keep in mind that, the :math:`\chi^2` (thus the
                        posterior) is summarized over ALL data. Thus, if using
                        mini-batch technique, the ratio of batch-size and of
                        data-size (we call `batch_ratio`) be introduced in the
                        computation of `chi_square_on_data`.

                    TODO:
                        What if the data-size is uncertain? For instance, when
                        training online, wherein the data-size is always keeping
                        increasing.

                    Args:
                        theta: `Tensor` with shape `[self.get_dim()]` and dtype
                               `self._float`.

                    Returns:
                        `Tensor` with shape `[]` and dtype `self._float`.
                    """
                    model = self.get_model()
                    _chi_square_on_batch = self._chi_square(
                        model=model, data_x=self.x, data_y=self.y,
                        data_y_error=self.y_error, params=theta)
                    # C.f. the "CAUTION" in docstring
                    _chi_square_on_data = tf.multiply( 1 / self.batch_ratio,
                                                      _chi_square_on_batch)
                    return _chi_square_on_data


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
                        being `self.num_samples`).
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


            with tf.name_scope('cat_gauss_mix_dist'):


                with tf.name_scope('categorical'):

                    self.weight = tf.nn.softmax(self.a,
                                                name='weight')

                    cat = Categorical(probs=self.weight,
                                      name='cat')


                with tf.name_scope('Gaussian'):

                    mu_list = tf.unstack(self.mu, name='mu_list')
                    zeta_list = tf.unstack(self.zeta, name='zeta_list')
                    components = [
                        MultivariateNormalDiagWithSoftplusScale(
                            loc=mu_list[i],
                            scale_diag=zeta_list[i],
                            name='Gaussian_{0}'.format(i))
                        for i in range(self.get_num_peaks())
                    ]


                with tf.name_scope('mixture'):

                    self.cat_gauss_mix_dist = Mixture(
                        cat=cat,
                        components=components,
                        name='cat_gauss_mix_dist')


            with tf.name_scope('loss'):

                theta_samples = self.cat_gauss_mix_dist.sample(
                    self.num_samples,
                    name='theta_samples')
                elbo = entropy.elbo_ratio(
                    log_p, self.cat_gauss_mix_dist, z=theta_samples,
                    name='ELBO')

                # Try Renyi divergence
                #elbo = entropy.renyi_ratio(
                #    log_p, self.cat_gauss_mix_dist,
                #    alpha=0.99, z=theta_samples,
                #    name='ELBO')

                # In TensorFlow, ELBO is defined as `E_q [ log( p / q ) ]`,
                # rather than as E_q [log(q / p)] as WikiPedia. So, the loss
                # would be `- elbo`
                self.loss = - elbo


            with tf.name_scope('optimize'):

                self.optimize = optimizer(self.learning_rate).minimize(self.loss)


            with tf.name_scope('model_prediction'):

                def _model_output(theta):
                    model = self.get_model()
                    return model(self.x, theta)

                with tf.name_scope('with_mean'):
                    self.model_prediction = tf.reduce_mean(
                        tf.map_fn(_model_output,
                                  theta_samples),
                        axis=0)

                # Someone may wonder
                with tf.name_scope('without_mean'):
                    self.model_prediction_without_mean = tf.map_fn(
                        _model_output,
                        theta_samples)


            with tf.name_scope('auxiliary_ops'):

                with tf.name_scope('metrics'):

                    with tf.name_scope('errror'):  # test!

                        mistakes = tf.not_equal(
                            tf.argmax(self.model_prediction, 1),
                            tf.argmax(self.y, 1),
                            name='mistakes')
                        self.error = tf.reduce_mean(
                            tf.cast(mistakes, self.get_var_dtype()),
                            name='error')



                with tf.name_scope('summarizer'):

                    # For loss
                    tf.summary.scalar('loss', self.loss)
                    tf.summary.histogram('loss', self.loss)

                    # For variable `a`
                    a_comps = tf.unstack(self.a)
                    for i, a_i in enumerate(a_comps):
                        tf.summary.scalar('a_{0}'.format(i),
                                          a_i)

                    # For variable `mu` (projected)  # test!
                    project_axis = 0
                    mu_comps = tf.unstack(self.mu)
                    for i, mu_comp in enumerate(mu_comps):
                        mu_sub_comps = tf.unstack(mu_comp)
                        tf.summary.scalar(
                            'mu_{0}_{1}'.format(i, project_axis),
                             mu_sub_comps[project_axis])

                    # It seems that, up to TF version 1.3, `tensor_summary` is
                    # still under building
                    #tf.summary.tensor_summary('a', self.a)
                    #tf.summary.tensor_summary('mu', self.mu)
                    #tf.summary.tensor_summary('zeta', self.zeta)

                    # For error, test!
                    tf.summary.scalar('error', self.error)
                    tf.summary.histogram('error', self.error)

                    # Merge them all
                    self.summary = tf.summary.merge_all()


                with tf.name_scope('initializer'):

                    self._initializer = tf.global_variables_initializer()
                    self._saver = tf.train.Saver()



        print('INFO - Compiled.')


    def fit(self,
            batch_generator,
            epochs,
            learning_rate,
            batch_ratio,
            num_samples=10**2,
            logdir=None,
            dir_to_ckpt=None,
            skip_steps=100):
        """
        TODO: complete docstring.
        TODO: add `tf.train.Saver()`, `global_step`, etc.
        """

        if logdir is not None:
            self._writer = tf.summary.FileWriter(logdir, self.graph)


        sess = self.get_session()
        saver = self.get_saver()
        initializer = self.get_initializer()

        with sess.as_default():

            sess.run(initializer)

            # -- Resotre from checkpoint
            initial_step = 0
            if dir_to_ckpt is not None:
                ckpt = tf.train.get_checkpoint_state(dir_to_ckpt)

                if ckpt and ckpt.model_checkpoint_path:
                    try:  # test!
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        initial_step = int(ckpt.model_checkpoint_path\
                                        .rsplit('-', 1)[1]) - 1
                        print('Restored from checkpoint at global step {0}'\
                            .format(initial_step+1))
                    except Exception as e:
                        print('ERROR - {0}'.format(e))
                        print('WARNING - Continue without restore.')


            # -- Iterating optimizer
            for step in range(initial_step, initial_step+epochs):

                x, y, y_error = next(batch_generator)
                feed_dict = {
                    self.x: x,
                    self.y: y,
                    self.y_error: y_error,
                    self.learning_rate: learning_rate,
                    self.batch_ratio: batch_ratio,
                    self.num_samples: num_samples,
                }


                # Write to `tensorboard`
                if logdir is None:
                    _, loss_val = sess.run(
                            [self.optimize, self.loss],
                            feed_dict=feed_dict)

                else:
                    _, loss_val, summary_val = sess.run(
                            [self.optimize, self.loss, self.summary],
                            feed_dict=feed_dict)
                    self._writer.add_summary(summary_val, global_step=step)


                # Save checkpoint
                if dir_to_ckpt is not None:
                    path_to_ckpt = os.path.join(dir_to_ckpt, 'checkpoint')
                    if (step+1) % skip_steps == 0:
                        saver.save(sess, path_to_ckpt, global_step=step+1)


            # -- Update the values of varialbes of `cat_gauss_mix_dist`
            self._a_val, self._mu_val, self._zeta_val = \
                sess.run([self.a, self.mu, self.zeta])

            # -- For visualization
            self._weight_val, self._sigma_val = \
                sess.run([self.weight, tf.nn.softplus(self.zeta)])



    def predict(self, x, num_samples=10**2, with_mean=True):
        """
        TODO: complete this docstring.
        """

        sess = self.get_session()

        with sess.as_default():

            feed_dict = {
                self.x: x,
                self.num_samples: num_samples,
            }

            if with_mean:
                predict_op = self.model_prediction
            else:
                predict_op = self.model_prediction_without_mean

            output_val = sess.run(predict_op,
                                  feed_dict=feed_dict)

        return output_val


    def finalize(self):
        """ Release the deployed resource by TensorFlow. """

        try:
            # Write the summaries to disk
            self._writer.flush()
            # Close the SummaryWriter
            self._writer.close()
        except:
            print('INFO - finalizing: no `SummaryWriter` to close.')
        # Close the Session
        self._sess.close()



    # -- Helper-Functions


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

        predict_y = model(data_x, params)
        noise = data_y - predict_y
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



    # -- Get-Functions

    def get_num_peaks(self):
        return self._num_peaks

    def get_dim(self):
        return self._dim

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
        """ Get the model (whose posteior is to be fitted by
            `cat_gauss_mix_dist`).

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

    def get_cat_gauss_mix_dist_params(self):
        """ Get tuple of numerical values (as numpy arraries) of
            `cat_gauss_mix_dist` parameters, including `weight`, `mu`,
            and `sigma`.

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

    def get_saver(self):
        """ Get the `tf.Saver()` within `self.graph`.

        Returns:
            `tf.Saver()` object.
        """
        return self._saver

    def get_initializer(self):
        """ Get the initializer within `self.graph`. """
        return self._initializer


    # -- Set-Functions
    def set_vars(self, vars_val):
        """ Set the values of variables of CMBD. This setting or re-setting can
            not work without re-calling the `self.compile()`.

        Args:
            vars_val:
                list of numpy array or `None`, as the values of the variables
                of `cat_gauss_mix_dist`.

        Modifies:
            `self._a_val`, `self._mu_val`, and `self._zeta_val`.
        """

        self._check_var_shape(*vars_val)

        # If passed this check without raising
        self._a_val, self._mu_val, self._zeta_val = vars_val
