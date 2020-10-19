# Auto-Surprise

![GitHub release (latest by date)](https://img.shields.io/github/v/release/BeelGroup/Auto-Surprise) ![PyPI](https://img.shields.io/pypi/v/Auto-Surprise.svg) [![Downloads](https://pepy.tech/badge/auto-surprise)](https://pepy.tech/project/auto-surprise) ![Codecov](https://img.shields.io/codecov/c/github/BeelGroup/Auto-Surprise.svg) ![Travis (.org)](https://img.shields.io/travis/BeelGroup/Auto-Surprise.svg)

Auto-Surprise is built as a wrapper around the Python [Surprise](https://surprise.readthedocs.io/en/stable/index.html) recommender-system library. It automates algorithm selection and hyper parameter optimization in a highly parallelized manner. 

- Documentation is available at [Auto-Surprise ReadTheDocs](https://auto-surprise.readthedocs.io/en/stable/)
- AutoSurprise is currently in development.

# Setup

Auto-Surprise is easy to install with Pip. You will require Python>=3.6 installed on a linux system. Currently not supported in windows, but can be used using [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

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
best_algo, best_params, best_score, tasks = engine.train(
    data=data, 
    target_metric='test_rmse', 
    cpu_time_limit=60 * 60, 
    max_evals=100
)
```

In the above example, we first initialize the `Engine`. We then run `engine.train()` to begin training our model. To train the model we need to pass the following

- `data` : The data as an instance of `surprise.dataset.DatasetAutoFolds`. Please read [Surprise Dataset docs](https://surprise.readthedocs.io/en/stable/dataset.html)
- `target_metric` : The metric we seek to minimize. Available options are `test_rmse` and `test_mae`.
- `cpu_time_limit` : The time limit we want to train. This is in seconds. For datasets like Movielens 100k, 1 hour is sufficient. But you may want to increase this based on the size of your dataset
- `max_evals`: The maximum number of evaluations each algorithm gets for hyper parameter optimization.
- `hpo_algo`: Auto-Surprise uses Hyperopt for hyperparameter tuning. By default, it's set to use TPE, but you can change this to any algorithm supported by hyperopt, such as Adaptive TPE or Random search.

## Setting the Hyperparameter Optimization Algorithm

Auto-Surprise uses Hyperopt. You can change the HPO algo as shown below.

```python
# Example for setting the HPO algorithm to adaptive TPE
import hyperopt

...

engine = Engine(verbose=True)
engine.train(
    data=data,
    target_metric='test_rmse',
    cpu_time_limit=60 * 60,
    max_evals=100,
    hpo_algo=hyperopt.atpe.suggest
)
```

## Building back the best model

You can build a pickelable model as shown.

```python
model = engine.build_model(best_algo, best_params)
```

# Benchmarks

In my testing, Auto-Surprise performed anywhere from 0.8 to 4% improvement in RMSE compared to the best performing default algorithm configuration. In the table below are the results for the Jester 2 dataset. Benchmark results for Movielens and Book-Crossing dataset are also available [here](https://auto-surprise.readthedocs.io/en/stable/benchmarks/results.html)

|       Algorithm      |  RMSE  |   MAE  |   Time   |
|:--------------------:|:------:|:------:|:--------:|
| Normal Predictor     |  7.277 |  5.886 | 00:00:01 |
| SVD                  |  4.905 |  3.97  | 00:00:13 |
| SVD++                |  5.102 |  4.055 | 00:00:29 |
| NMF                  |   --   |   --   |    --    |
| Slope One            |  5.189 |  3.945 | 00:00:02 |
| KNN Basic            |  5.078 |  4.034 | 00:02:14 |
| KNN with Means       |  5.124 |  3.955 | 00:02:16 |
| KNN with   Z-score   |  5.219 |  3.955 | 00:02:20 |
| KNN Baseline         |  4.898 |  3.896 | 00:02:14 |
| Co-clustering        |  5.153 |  3.917 | 00:00:12 |
| Baseline Only        |  4.849 |  3.934 | 00:00:01 |
| GridSearch           | 4.7409 | 3.8147 | 80:52:35 |
| Auto-Surprise (TPE)  | 4.6489 | 3.6837 | 02:00:10 |
| Auto-Surprise (ATPE) | 4.6555 | 3.6906 | 02:00:01 |

# Papers

[Auto-Surprise: An Automated Recommender-System (AutoRecSys) Library with Tree of Parzens Estimator (TPE) Optimization](https://dl.acm.org/doi/abs/10.1145/3383313.3411467?casa_token=ADmaOhK2tHgAAAAA:4UXHmuLXM_gJYQdUZp7ab5hwn-eNv2Daot5FtfYLG3m1KYLc99Y1_rhwzY2qcCJySUhoFBAfGnt5Qg)
