"""
Testing gridsearch for ml-100k dataset using same defined search space
Algorihm used is KNNWithMeans as it showed the best performance.
"""

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
from surprise.model_selection import GridSearchCV

import time
import datetime
import os
import numpy as np
import pandas as pd
import sys

sys.path.insert(1, './')

from auto_surprise.engine import Engine

if __name__ == '__main__':
    print("Starting benchmark")

    # Load Movielens 100k dataset Dataset
    file_path = os.path.expanduser('../datasets/ml-100k/u.data')
    reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(1, 5))

    data = Dataset.load_from_file(file_path, reader=reader)

    benchmark_results = {
        'Algorithm': [],
        'RMSE': [],
        'MAE': [],
        'Time': [],
        'Best params': []
    }

    SIMILARITY_OPTIONS_SPACE = {
        'name': ['cosine', 'msd', 'pearson', 'pearson_baseline'],
        'user_based': [False, True],
        'shrinkage': range(1, 300)
    }

    SPACE = {
        'k': range(1,100),
        'min_k': range(1,10),
        'sim_options': SIMILARITY_OPTIONS_SPACE
    }

    start_time = time.time()

    gs = GridSearchCV(KNNWithMeans, SPACE, measures=['rmse', 'mae'], cv=5)

    gs.fit(data)
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))

    # best RMSE score
    print(gs.best_score['rmse'])

    benchmark_results['Algorithm'].append('KNN Gridsearch')
    benchmark_results['RMSE'].append(gs.best_score['rmse'])
    benchmark_results['MAE'].append(gs.best_score['mae'])
    benchmark_results['Time'].append(cv_time)
    benchmark_results['Best params'].append(gs.best_params)

    # Load results to csv
    results = pd.DataFrame.from_dict(benchmark_results)
    print(results)
    results.to_csv('knn-gridsearch.csv')
