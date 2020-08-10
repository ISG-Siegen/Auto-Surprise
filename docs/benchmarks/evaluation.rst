.. benchmarks/_evaluation
.. _evaluation:

Evaluation
==========

We tested Auto-Surprise against 3 datasets

- Movielens 100k
- Jester Dataset 2 (100k Random sample)
- Book Crossing (100k random sample)

We then ran all surprise algorithms in their default configuration. We then ran Auto-Surprise with a time limit set to 2 hours and the target metric as RMSE. We also compared our results to gridsearch on a smaller search space. 
