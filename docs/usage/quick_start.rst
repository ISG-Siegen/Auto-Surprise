.. usage/_quick_start
.. _quick_start:

Quick Start
===========

Installing
..........

You will require Python >=3.6 and a linux based OS. With pip, installing Auto-Surprise is as easy as

.. parsed-literal::

    pip install auto-surprise

Thats it. You are ready to get started

Quick Example
.............

Here's a quick example of using Auto-Surprise to determine the best algorithm and hyperparameters for the Movielens 100k dataset. 

.. code-block:: python

    # Import required libraries
    from surprise import Dataset
    from auto_surprise.engine import Engine

    # Load the dataset
    data = Dataset.load_builtin('ml-100k')

    # Intitialize auto surprise engine
    engine = Engine(debug=False)

    # Start the trainer
    best_algo, best_params, best_score, tasks = engine.train(
        data=data, 
        target_metric='test_rmse', 
        cpu_time_limit=60 * 60, # Run for 1 hour 
        max_evals=100
    )

Thats it, after about 1 hour you should have the best algorithm along with the best parameters. To learn more, continue with the :ref:`manual`