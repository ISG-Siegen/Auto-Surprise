"""
Testing gridsearch for ml-100k dataset using same defined search space
Algorihm used is KNNWithMeans as it showed the best performance.
"""

import sys
import time
import datetime
import os
import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

sys.path.insert(1, './')

from auto_surprise.algorithms.auto_surprise_knn_with_means import AutoSurpriseKNNWithMeans

if __name__ == '__main__':
    print("Starting benchmark")

    # Load Movielens 100k dataset Dataset
    file_path = os.path.expanduser('../datasets/ml-100k/u.data')
    reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(1, 5))

    data = Dataset.load_from_file(file_path, reader=reader)

    benchmark_results = {
        'Algorithm': [],
        'RMSE': [],
        'Time': [],
        'Best params': []
    }

    for i in range(0, 100):
        print("Iteration : " + i)
        start_time = time.time()

        algo = AutoSurpriseKNNWithMeans(data=data)
        best_hyperparams, trials = algo.best_hyperparams(max_evals=20)

        best_trial = None
        best_trial = sorted(trials.results, key=lambda x: x['loss'], reverse=False)[0]

        cv_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))

        benchmark_results['Algorithm'].append('KNN Hyperopt')
        benchmark_results['RMSE'].append(best_trial['loss'])
        benchmark_results['Time'].append(cv_time)
        benchmark_results['Best params'].append(best_hyperparams)

    # Load results to csv
    results = pd.DataFrame.from_dict(benchmark_results)
    print(results)
    results.to_csv('knn-hyperopt.csv')
