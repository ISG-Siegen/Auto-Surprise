.. Auto-Surprise documentation master file, created by
   sphinx-quickstart on Wed Jun 17 18:18:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Auto-Surprise's documentation!
=========================================

Auto-Surprise is an easy-to-use python AutoRecommenderSystem (AutoRecSys) library. It automates algorithm selection and hyperparameter tuning to build an optimized recommendation model.
It uses the popular scikit library `Surprise <http://surpriselib.com/>`_ for recommender algorithms and `Hyperopt <https://github.com/hyperopt/hyperopt>`_ for hyperparameter tuning.

Unfortunately, currently only linux systems are supported, but you can use WSL in windows as well.

To get started with Auto-Surprise, check out the :ref:`quick_start` guide. If you have any issues or doubts, head over to the `Github repository <https://github.com/BeelGroup/Auto-Surprise>`_ and create an issue.

.. toctree::
   :maxdepth: 3
   :caption: Usage Guide
   
   usage/quick_start
   usage/manual
   usage/reproducing_experiments

.. toctree::
   :caption: Benchmarks

   benchmarks/evaluation
   benchmarks/results
