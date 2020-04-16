# AutoSurprise

Auto-Surprise is built as a wrapper around the Python [Surprise](https://surprise.readthedocs.io/en/stable/index.html) recommender-system library. It automates algorithm selection and hyper parameter optimization in a highly parallelized manner.

AutoSurprise is currently in development.

# Setup

To setup AutoSurprise, you will require Python3 installed on a linux system.

To install the required packages

```
pip install -r requirements.txt
```

You can then test that the install works by running the example program

```
python3 -u examples/ml-100k.py
```

# Usage

Basic usage of AutoSurprise is given below.

```python
from surprise import Dataset
from auto_surprise.engine import Engine

# Load the dataset
if __name__ == '__main__':
    data = Dataset.load_builtin('ml-100k')

    # Intitialize auto surprise engine
    engine = Engine(debug=False)

    # Start the trainer
    best_model, best_params, best_score, tasks = engine.train(data=data, target_metric='test_rmse', cpu_time_limit=720, max_evals=100)
```

In the above example, we first initialize the `Engine`. We then run `engine.train()` to begin training our model. To train the model we need to pass the following

- `data` : The data as an instance of `surprise.dataset.DatasetAutoFolds`. Please read [Surprise Dataset docs](https://surprise.readthedocs.io/en/stable/dataset.html)
- `target_metric` : The metric we seek to minimize. Available options are `test_rmse` and `test_mae`.
- `cpu_time_limit` : The time limit we want to train. This is in seconds. For datasets like Movielens 100k, 1 hour is sufficient. But you may want to increase this based on the size of your dataset
- `max_evals`: The maximum number of evaluations each algorithm gets for hyper parameter optimization.
