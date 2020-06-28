.. usage/_manual
.. _manual:

Manual
======

Here, we will cover in more detail the usage for Auto-Surprise. We will start with an example, and go through each section

.. code-block:: python

    import hyperopt
    from surprise import Reader, Dataset
    from auto_surprise.engine import Engine

    # Load the movielens dataset
    file_path = os.path.expanduser('./ml-100k/u.data')
    reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(1, 5))
    data = Dataset.load_from_file(file_path, reader=reader)

    # Intitialize auto surprise engine
    engine = Engine(verbose=True)

    # Start the trainer
    best_algo, best_params, best_score, tasks = engine.train(
        data=data, 
        target_metric='test_rmse', 
        cpu_time_limit=60*60*2, 
        max_evals=100,
        hpo_algo=hyperopt.tpe.suggest
    )

    # Build the model using the best algorithm and hyperparameters
    best_model = engine.build_model(best_algo, best_params)

Loading the dataset
...................

Auto-Surprise requires your dataset to be an instance of `surprise.dataset.DatasetAutoFolds`. You can learn more about this by reading the `Surprise Dataset Docs <https://surprise.readthedocs.io/en/stable/dataset.html>`_

Initializing Auto-Surprise Engine
.................................

`Engine` is the main class for Auto-Surprise. You will need to initialize it once before you start using it.

.. code-block:: python

    engine = Engine(verbose=True, algorithms=['svd', 'svdpp', 'knn_basic', 'knn_baseline'])

* `verbose` : By default set to `True`. Controls the verbosity of Auto-Surprise.
* `algorithms` : The algorithms to be optimized. Must be in the form of an array of strings. Available choices are `['svd', 'svdpp', 'nmf', 'knn_basic', 'knn_baseline', 'knn_with_means', 'knn_with_z_score', 'co_clustering', 'slope_one', 'baseline_only']` 

Starting the Optimization process
.................................

To start the optimization method, you can use the `train` method of `Engine`. This will return the best algorithm, hyperparameters, best score, and tasks completed.

.. code-block:: python

    best_algo, best_params, best_score, tasks = engine.train(
        data=data, 
        target_metric='test_rmse', 
        cpu_time_limit=60*60*2, 
        max_evals=100,
        hpo_algo=hyperopt.tpe.suggest
    )

There are a few parameters you can use.

* `data` : The data as an instance of `surprise.dataset.DatasetAutoFolds`.
* `target_metric` : The metric we seek to minimize. Available options are `test_rmse` and `test_mae`.
* `cpu_time_limit` : The time limit we want to train. This is in seconds. For datasets like Movielens 100k, 1-2 hours is sufficient. But you may want to increase this based on the size of your dataset
* `max_evals`: The maximum number of evaluations each algorithm gets for hyper parameter optimization.
* `hpo_algo`: Auto-Surprise uses Hyperopt for hyperparameter tuning. By default, it's set to use TPE, but you can change this to any algorithm supported by hyperopt, such as Adaptive TPE or Random search.

Building the best Model
.......................

You can use the best alogithm and best hypermaters you got from the `train` method to build a model.

.. code-block:: python
    
    best_model = engine.build_model(best_algo, best_params)

You can pickle this model to save it and use it elsewhere.
