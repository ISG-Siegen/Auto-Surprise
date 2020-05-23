import time
import datetime
import os
import sys
from surprise import Dataset
from auto_surprise.engine import Engine

if __name__ == '__main__':

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
