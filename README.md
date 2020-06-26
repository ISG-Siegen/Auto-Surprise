# Auto-Surprise

![GitHub release (latest by date)](https://img.shields.io/github/v/release/BeelGroup/Auto-Surprise.svg) ![PyPI](https://img.shields.io/pypi/v/Auto-Surprise.svg) ![PyPI - Downloads](https://img.shields.io/pypi/dm/Auto-Surprise.svg) ![Codecov](https://img.shields.io/codecov/c/github/BeelGroup/Auto-Surprise.svg) ![Travis (.org)](https://img.shields.io/travis/BeelGroup/Auto-Surprise.svg)

Auto-Surprise is built as a wrapper around the Python [Surprise](https://surprise.readthedocs.io/en/stable/index.html) recommender-system library. It automates algorithm selection and hyper parameter optimization in a highly parallelized manner. Full documentation is available at [Auto-Surprise ReadTheDocs](https://auto-surprise.readthedocs.io/en/stable/)

AutoSurprise is currently in development.

# Setup

To setup Auto-Surprise, you will require Python3 installed on a linux system. Auto-Surprise can be installed using pip

```bash
$ pip install auto-surprise
```

# Usage

Basic usage of AutoSurprise is given below.

```python
from surprise import Dataset
from auto_surprise.engine import Engine

# Load the dataset
data = Dataset.load_builtin('ml-100k')

# Intitialize auto surprise engine
engine = Engine(verbose=True)

# Start the trainer
best_algo, best_params, best_score, tasks = engine.train(data=data, target_metric='test_rmse', cpu_time_limit=720, max_evals=100)
```

In the above example, we first initialize the `Engine`. We then run `engine.train()` to begin training our model. To train the model we need to pass the following

- `data` : The data as an instance of `surprise.dataset.DatasetAutoFolds`. Please read [Surprise Dataset docs](https://surprise.readthedocs.io/en/stable/dataset.html)
- `target_metric` : The metric we seek to minimize. Available options are `test_rmse` and `test_mae`.
- `cpu_time_limit` : The time limit we want to train. This is in seconds. For datasets like Movielens 100k, 1 hour is sufficient. But you may want to increase this based on the size of your dataset
- `max_evals`: The maximum number of evaluations each algorithm gets for hyper parameter optimization.
- `hpo_algo`: Auto-Surprise uses Hyperopt for hyperparameter tuning. By default, it's set to use TPE, but you can change this to any algorithm supported by hyperopt, such as Adaptive TPE or Random search.

```python
# Example for setting the HPO algorithm to adaptive TPE
import hyperopt

...

engine = Engine(verbose=True)
engine.train(
    data=data,
    target_metric='test_rmse',
    cpu_time_limit=720,
    max_evals=100,
    hpo_algo=hyperopt.atpe.suggest
)
```
