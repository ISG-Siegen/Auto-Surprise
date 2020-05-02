from surprise import Dataset
from surprise import Reader
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
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
import numpy as np
import pandas as pd
import sys

sys.path.insert(1, './')

from auto_surprise.engine import Engine

if __name__ == '__main__':
    print("Starting benchmark")
    # Surprise algorithms to evaluate
    algorithms = (SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, CoClustering, BaselineOnly, NormalPredictor)

    # Load Jester dataset 2
    file_path = os.path.expanduser('../datasets/jester_dataset_2/jester_ratings.dat')
    reader = Reader(line_format='user item rating', rating_scale=(-10, 10), sep="\t\t")

    data = Dataset.load_from_file(file_path, reader=reader)

    benchmark_results = {
        'Algorithm': [],
        'RMSE': [],
        'MAE': [],
        'Time': []
    }

    # Evaluate Surprise Algorithms
    for algo in algorithms:
        algo_name = algo.__name__

        print("Running algorithm : %s" % algo_name)

        try:
            start_time = time.time()

            cv_results = cross_validate(algo(), data, ['rmse', 'mae'])

            cv_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            mean_rmse = '{:.3f}'.format(np.mean(cv_results['test_rmse']))
            mean_mae = '{:.3f}'.format(np.mean(cv_results['test_mae']))

            benchmark_results['Algorithm'].append(algo_name)
            benchmark_results['RMSE'].append(mean_rmse)
            benchmark_results['MAE'].append(mean_mae)
            benchmark_results['Time'].append(cv_time)
        except Exception as e:
            print('Exception : ', e)

    # Evaluate AutoSurprise
    start_time = time.time()
    engine = Engine(debug=False)
    time_limt = 691200 # Run for 8 days
    best_model, best_params, best_score, tasks = engine.train(data=data, target_metric='test_rmse', quick_compute=False, cpu_time_limit=time_limt, max_evals=1000000)

    cv_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    cv_results = cross_validate(engine.build_model(best_model, best_params), data, ['rmse', 'mae'])
    mean_rmse = '{:.4f}'.format(np.mean(cv_results['test_rmse']))
    mean_mae = '{:.4f}'.format(np.mean(cv_results['test_mae']))

    print("--------- Done ----------")
    print("Best model: ", best_model)
    print("Best params: ", best_params)
    print("Best score: ", best_score)
    print("All tasks: ", tasks)


    benchmark_results['Algorithm'].append('AutoSurprise')
    benchmark_results['RMSE'].append(mean_rmse)
    benchmark_results['MAE'].append(mean_mae)
    benchmark_results['Time'].append(cv_time)

    # Load results to csv
    results = pd.DataFrame.from_dict(benchmark_results)
    print(results)
    results.to_csv('jester-benchmark-results.csv')
