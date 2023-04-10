.. usage/_reproducing_experiments
.. _reproducing_experiments:

Reproducing Experiments
=======================

You may want to make sure that your results are reproducible. This can be done easily by setting the seed and random state when initializing `Engine`.

.. code-block:: python
    import numpy
    from auto_surprise.engine import Engine

    # Intitialize auto surprise engine with random state set
    engine = Engine(verbose=True, random_state=numpy.random.default_rng(12345))

This will make sure that you're results will be exactly the same, provided you're other training params also stay the same.

