# AutoSurprise

Auto-Surprise is built as a wrapper around the Python [Surprise](https://surprise.readthedocs.io/en/stable/index.html) recommender-system library. Auto-Surprise utilizes Bayesian Optimization for the algorithm selection and configuration, and brings the advances of AutoML to the recommender-system community

AutoSurprise is currently in development.

# Usage

Basic usage of AutoSurprise is given below. Note that to use AutoSurprise, your data must be an instance of `surprise.dataset.DatasetAutoFolds`. Please read [Surprise Dataset docs](https://surprise.readthedocs.io/en/stable/dataset.html)

```python
from surprise import Dataset
from auto_surprise.engine import Engine

# Load the dataset
data = Dataset.load_builtin('ml-100k')

# Intitialize auto surprise engine
engine = Engine(debug=False)

# Start the trainer
best_model, best_params, best_score, tasks = engine.train(data=data, target_metric='test_rmse')
```
