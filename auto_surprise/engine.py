from auto_surprise.constants import ALGORITHM_MAP, DEFAULT_TARGET_METRIC, MAX_WORKERS, DEFAULT_MAX_EVALS, FULL_ALGO_LIST, QUICK_COMPUTE_ALGO_LIST
from auto_surprise.trainer import Trainer
from auto_surprise.exceptions import ValidationError
import auto_surprise.validation_util as validation_util

from surprise.model_selection import KFold
import concurrent.futures

class Engine(object):
    def __init__(self, debug=False):
        """
        Initialize new engine
        """
        self._debug = debug

    def train(self, target_metric=DEFAULT_TARGET_METRIC, data=None, max_evals=DEFAULT_MAX_EVALS, quick_compute=False):
        """
        Train and find most optimal model and hyperparameters
        """
        try:
            # Validations
            validation_util.validate_target_metric(target_metric)
            validation_util.validate_dataset(data)
            validation_util.validate_max_evals(max_evals)

            tasks = {}
            algorithms = QUICK_COMPUTE_ALGO_LIST if quick_compute else FULL_ALGO_LIST
            iteration = 0
            while True:
                print("Iteration: %i" % iteration)

                tasks[iteration] = {}
                futures = {}
                with concurrent.futures.ProcessPoolExecutor(max_workers = MAX_WORKERS) as executor:
                    # Run for N algorithms
                    for algo in algorithms:
                        if self._debug:
                            print("Starting thread with %s algorithm" % algo)

                        trainer = Trainer(algo=algo, data=data, target_metric=target_metric, debug=self._debug)
                        futures[executor.submit(trainer.start)] = algo

                    for future in concurrent.futures.as_completed(futures):
                        algo = futures[future]
                        tasks[iteration][algo] = future.result()

                if len(tasks[iteration]) == 1:
                    break
                else:
                    """
                    Rank N algorithms and take the top N/2 algorithms for the next iteration
                    """
                    algorithms_ranking = [i[0] for i in sorted(tasks[iteration].items(), key=lambda x: x[1][1]['loss'], reverse=False)]
                    # TODO: Think of a better alternative for this
                    algorithms_count = round(len(algorithms) / 2)
                    algorithms = algorithms_ranking[0:algorithms_count]
                    iteration += 1

            best_model = list(tasks[iteration].keys())[0]
            best_params = tasks[iteration][best_model][1]
            best_score = tasks[iteration][best_model][1]['loss']

            return best_model, best_params, best_score, tasks
        except ValidationError as e:
            """
            Catch validation errors
            """
            print(e.message)
            raise
