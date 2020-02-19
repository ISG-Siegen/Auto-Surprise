"""
Benchmarks for the dating dataset

@inproceedings{brozovsky07recommender,
    author = {Lukas Brozovsky and Vaclav Petricek},
    title = {Recommender System for Online Dating Service},
    booktitle = {Proceedings of Conference Znalosti 2007},
    year = {2007},
    isbn = {},
    pages = {},
    url = {http://www.occamslab.com/petricek/papers/dating/brozovsky07recommender.pdf},
    location = {Ostrava, Czech Republic},
    publisher = {VSB},
    address = {Ostrava},
}
"""

import sys
import time
import datetime
import os
import numpy as np
import pandas as pd

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

sys.path.insert(1, './')

from auto_surprise.engine import Engine

if __name__ == '__main__':
    print("Starting benchmark")
    # Surprise algorithms to evaluate
    algorithms = (SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline, CoClustering, BaselineOnly, NormalPredictor)

    # Load dataset
    file_path = os.path.expanduser('../datasets/libimseti/ratings.dat')
    reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(1, 10))
    data = Dataset.load_from_file(file_path, reader=reader)

    benchmark_results = {
        'Algorithm': [],
        'RMSE': [],
        'MAE': [],
        'Best params': [],
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
            benchmark_results['Best params'].append({})
            benchmark_results['Time'].append(cv_time)

        except Exception as exc:
            print('Exception : ', exc)

    # Evaluate AutoSurprise without SVD++
    start_time = time.time()
    engine = Engine(debug=False)
    best_model, best_params, best_score, tasks = engine.train(data=data, target_metric='test_rmse', quick_compute=True)
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))

    print("--------- Done ----------")
    print("Best model: ", best_model)
    print("Best params: ", best_params)
    print("Best score: ", best_score)
    print("All tasks: ", tasks)

    benchmark_results['Algorithm'].append('AutoSurprise (No SVD++)')
    benchmark_results['RMSE'].append(best_score)
    benchmark_results['MAE'].append(best_score)
    benchmark_results['Best params'].append(best_params)
    benchmark_results['Time'].append(cv_time)

    # Evaluate AutoSurprise
    start_time = time.time()
    best_model, best_params, best_score, tasks = engine.train(data=data, target_metric='test_rmse', quick_compute=False)
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))

    print("--------- Done ----------")
    print("Best model: ", best_model)
    print("Best params: ", best_params)
    print("Best score: ", best_score)
    print("All tasks: ", tasks)


    benchmark_results['Algorithm'].append('AutoSurprise')
    benchmark_results['RMSE'].append(best_score)
    benchmark_results['MAE'].append(best_score)
    benchmark_results['Best params'].append(best_params)
    benchmark_results['Time'].append(cv_time)

    # Load results to csv
    results = pd.DataFrame.from_dict(benchmark_results)
    print(results)
    results.to_csv('dating-dataset-result.csv')
