from surprise import Dataset
from surprise import Reader
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import KNNWithZScore
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise.model_selection import cross_validate
import hyperopt
import time
import datetime
import os
import numpy as np
import pandas as pd
import sys

sys.path.insert(1, './')

from auto_surprise.engine import Engine

if __name__ == '__main__':
    sys.settrace
    print("Starting benchmark")
    # Surprise algorithms to evaluate
    algorithms = (SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, CoClustering, BaselineOnly, NormalPredictor)

    # Load Book crossing dataset
    df = pd.read_csv('../datasets/BX-CSV-DUMP/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    df.columns = ['user', 'item', 'rating']

    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=134), reader=reader)
    del(df)

    benchmark_results = {
        'Algorithm': [],
        'RMSE': [],
        'MAE': [],
        'Time': []
    }

    # Evaluate AutoSurprise
    start_time = time.time()
    time_limt = 60 * 60 * 12 # Run for 12 hours
    engine = Engine(verbose=False)
    best_model, best_params, best_score, tasks = engine.train(data=data, target_metric='test_rmse', quick_compute=False, cpu_time_limit=time_limt, max_evals=10000, hpo_algo=hyperopt.atpe.suggest)

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

    print("--- AutoSurprise results ---")
    print(pd.DataFrame.from_dict(benchmark_results))

    # Evaluate Surprise Algorithms
    for algo in algorithms:
        algo_name = algo.__name__

        print("Running algorithm : %s" % algo_name)

        try:
            start_time = time.time()

            cv_results = cross_validate(algo(), data, ['rmse', 'mae'], cv=3)

            cv_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            mean_rmse = '{:.4f}'.format(np.mean(cv_results['test_rmse']))
            mean_mae = '{:.4f}'.format(np.mean(cv_results['test_mae']))

            benchmark_results['Algorithm'].append(algo_name)
            benchmark_results['RMSE'].append(mean_rmse)
            benchmark_results['MAE'].append(mean_mae)
            benchmark_results['Time'].append(cv_time)

        except Exception as e:
            print('Exception : ', e)

    # Load results to csv
    results = pd.DataFrame.from_dict(benchmark_results)
    print(results)
    results.to_csv('book-crossing-benchmar-results.csv')
