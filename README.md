Neural Network for Posterior
============================


TODO
----


[ ] - Write an abstract `Trainer()` class for `fit()`.
    
  * Set `learning_rate` as a placeholder.
    
  * Try to adjust learning-rate (**not** wihtin optimizer) automatically,
    basing on the suggestions in _Deep Learning_ book, chapter _Methodology_.

[ ] - Use `tfdbg` on `nn4post/tests/shadow_neural_network` for the strange
      `ERROR` of `DynamicPartition`.

[ ] - Try to visuralize the weights, gradients, etc, to see why it get stuck,
      even though being more representable (with more peaks).
