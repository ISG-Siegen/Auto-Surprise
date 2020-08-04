.. usage/_reproducing_experiments
.. _reproducing_experiments:

Reproducing Experiments
=======================

You may want to make sure that your results are reproducible. This can be done easily by setting the seed and random state when initializing `Engine`.

.. code-block:: python

    from auto_surprise.engine import Engine

    random.seed(123)
    numpy.random.seed(123)

    # Intitialize auto surprise engine with random state set
    engine = Engine(verbose=True, random_state=numpy.random.RandomState(123))

This will make sure that you're results will be exactly the same, provided you're other training params also stay the same.
