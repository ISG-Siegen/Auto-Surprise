from surprise import Dataset
from surprise import Reader
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise.model_selection import cross_validate

import time
import datetime
import os
import sys

sys.path.insert(1, './')

from auto_surprise.engine import Engine

if __name__ == '__main__':
    # Load Movielens 100k dataset Dataset
    # file_path = os.path.expanduser('../datasets/ml-100k/u.data')
    # reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(1, 5))

    data = Dataset.load_builtin('ml-100k')

    # Run auto surprise
    start_time = time.time()
    engine = Engine(debug=False)
    best_model, best_params, best_score, tasks=engine.train(data=data, target_metric='test_rmse', cpu_time_limit=720, max_evals=100)
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))

    print("--------- Done ----------")
    print("Time taken: ", cv_time)
    print("Best model: ", best_model)
    print("Best params: ", best_params)
    print("Best score: ", best_score)
    print("All tasks: ", tasks)
