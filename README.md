Neural Network for Posterior
============================


TODO
----


[ ] - Write an abstract `Trainer()` class for `fit()`.
    
  * Set `learning_rate` as a placeholder.
    
  * Try to adjust learning-rate (**not** wihtin optimizer) automatically,
    basing on the suggestions in _Deep Learning_ book, chapter _Methodology_.

[X] - Use `tfdbg` on `nn4post/tests/shallow_neural_network` for the strange
      `ERROR` of `DynamicPartition`.

[X] - Try to visuralize the weights, gradients, etc, to see why it get stuck,
      even though being more representable (with more peaks).

#[ ] - Write a parser that transforms the flatten model parameters back to the un-flatten.

[X] - Edward: write a template of variational inference with rich docstrings.
        
        See 'nn4post/sample/edward_template.py'.

[X] - Test shallow neural network on fitting `sin()`.

[X] - Test shallow nerual network on MNIST dataset, comparing with that by Nealson.

[ ] - Load and restore values of Variables, as numpy arraies, with Pickle.

[ ] - Write a `KLqp` that inherits `edward.Inference.VariationalInference`, by BayesFlow.

[ ] - And test this `KLqp` on MNIST dataset.
