Trials
=========

Logs the trials.


Comparisons
-------------


### Score on MNIST

* Edward on MNIST:

  - elapsed 1200s per epoch, reaching 95% on validation data after the first
    epoch.
  
  - Reaching 97.2% accuracy on test data after 100 epochs, with the same
    configuration proposed by Nealson, instead of being 98% as he said.

* MAP on MNIST:

  - elapsed 24s per epoch, reaching 85% on validation data after the first
    epoch.
    
    

### Elapsed Time

Without logging some irrelavent parameters, we get an interesting profilings,
**with the first line as the base of comparison for each case**:

  * `n_samples = 10`:

      - `edward` elapses `0.05`s for each iteration (except for the first).

      - `nn4post_ed` elapses `13.1`s for each iteration (except for the first)
        with `n_cats = 1`.

      - `nn4post_ed` elapses `12.6`s for each iteration (except for the first)
        with `n_cats = 10`.

      - `nn4post_ed` elapses `13.7`s for each iteration (except for the first)
        with `n_cats = 20`.

      - `nn4post_ed` elapses `14.3`s for each iteration (except for the first)
        with `n_cats = 100`.

  * `n_samples = 100`:

      - `edward` elapses `0.43`s for each iteration (except for the first).

      - `nn4post_ed` elapses `128`s for each iteration (except for the first)
        with `n_cats = 1`.

      - `nn4post_ed` elapses `141`s for each iteration (except for the first)
        with `n_cats = 10`.

      - `nn4post_ed` elapses `???`s for each iteration (except for the first)
        with `n_cats = 100`. (Being too slow in `KLqp.build_loss_and_gradients`
        (which is called in `KLqp.initialize`).)

Thus from these trials, we find the elapsed time:

  * is proportional to `n_samples`, as theoritically expected;
  
  * is quite in-sensitive to `n_cats`, to our surprise.
