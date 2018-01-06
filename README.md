Neural Network for Posterior
============================

C.f. the [documentation](https://github.com/shuiruge/nn4post/blob/master/docs/main.pdf).


Installation
----------


### INSTALL

First `cd` to the directory of this git repository, and then

    pip install .


### UNINSTALL

Directly,

    pip uninstall nn4post
    
    
    
HowTo
-----

If you feel the documentation is too long to be read, this section will give you
a shortcut.


### Define Posterior

First, define your posterior in its logorithm:

    def log_posterior(param_1, param_2, ...):
        """
        Args:
            param_1:
                An instance of `tf.Tensor`.
            param_2:
                An instance of `tf.Tensor`.
            ...
                
        Returns:
            Scalar.
        """
        # Your implementation.



### Vectorization

Then you have to vectorize your `log_posterior`. This is because that `nn4post`
is made for general `log_posterior`, regardless of its arragement of arguments,
thus works on the Euclidean parameter-space only. Even though, we have provided
helpful utils in `nn4post.utils` module for vectorization.

If you have known the shapes of the parameters in the argument `param` or the
arguments `param_i`s, then collect the shapes into a dictionary with keys the
names of parameter and values the associated shapes, like

    param_shape = {'param_1': [5, 10], 'param_2': [3], ...}
    
Then get the dimension of the Euclidean parameter-space by

    param_space_dim = nn4post.utils.get_param_space_dim(param_shape)

and directly vectorize your `log_posterior` by a decorator


    @nn4post.utils.vectorize(param_shape)
    def log_posterior(param_1, param_2, ...):
        # Your implementation.

And if you do not know the shapes of the parameters, but have known the priors
of the parameters as instances of `tf.distributions.Distribution`, or have known
some specific values of the parameters as `numpy.array`s or `tf.Tensor`s, then
collect them as a dictionary like

    param_val = {'param_1': val_1, 'param_2': val_2, ...}
    
where `val_i`s are either instances of `tf.distributions.Distribution`, or of
`numpy.array` or `tf.Tensor`, then obtain the shapes of the parameters, as
a dictionary, by simply

    param_shape = nn4post.utils.get_param_shape(param_val)
    
And then follow the previous to vectorize your `log_posterior`.


### Inference

First set argument `n_c`, which is the number of Gaussian distributions in the
mixture distribution, or say, the number of "perceptrons". Then set the arguement
`n_d`, which is the dimension of Euclidean parameter-space, and which we have
known, as the `param_shape_dim`.

Then calling `build_nn4post(N_C, param_space_dim, log_posterior)` will setup a
computational graph of TensorFlow, and return `collection` and `grads_and_vars`.
The `collection` is a dictionary contains "loss", etc. This is just for
convenience, since these are also involved in the collection of the TensorFlow
graph. The `grads_and_vars` is as the argument of the method
`tf.train.Optimizer.apply_gradients()`.

Then you can train the inference model by the standard process in TensorFlow
by `tf.train.Optimizer` in a session, say `sess`.


### Sampling

After training, read out the value of the trained variables of this inference
model by running

    trained_var = {}
    for var_name in ['a', 'mu', 'zeta']:
        var_value = .sess.run(collection[var_name])
        trained_var[var_name] = var_value
        
And then get the trained inference distribution by

    trained_q = nn4post.get_trained_q(trained_var)
    
which is an instance of `tf.distributions.Distribution`, thus from which we
can sample the paramters in the Euclidean parameter-space, say
`sampled_euclidean_param`. And to parse the Euclidean parameter to the parameter
we want, use the

    parse_param = nn4post.utils.get_parse_param(param_shape)
    
Thus

    sampled_param = parse_param(sampled_euclidean_param)


Q.E.D.
