Log of `nn4post` on MNIST Dataset
===============================================


* Increasing `n_peaks` from `1` to `5` does lowered the `error` (as metric), but
  lowered just a bit. This hints that the "geography" of the posterior of MNIST
  dataset with Gaussian prior has a single large peak with several smaller peaks
  around.

* Increasing `n_hidden` from `30` to `100` decreases `error` little.
