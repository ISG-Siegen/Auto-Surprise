from auto_surprise.constants import (ALGORITHM_MAP, DEFAULT_TARGET_METRIC, MAX_WORKERS,
                                    DEFAULT_MAX_EVALS, FULL_ALGO_LIST, QUICK_COMPUTE_ALGO_LIST,
                                    BASELINE_ALGO, EVALS_MULTIPLIER)
from auto_surprise.trainer import Trainer
from auto_surprise.exceptions import ValidationError
from auto_surprise.strategies.basic_reduction import BasicReduction
import auto_surprise.validation_util as validation_util

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

            # Determine baseline value from random normal predictor
            baseline_trainer = Trainer(algo=BASELINE_ALGO, data=data, target_metric=target_metric, debug=self._debug)
            baseline_loss = baseline_trainer.start(1)[1]['loss']

            tasks = {}
            strategy = BasicReduction()
            algorithms = QUICK_COMPUTE_ALGO_LIST if quick_compute else FULL_ALGO_LIST
            iteration = 0
            while True:
                print("Iteration: %i" % iteration)

                max_evals = EVALS_MULTIPLIER * (iteration + 1)
                tasks[iteration] = {}
                futures = {}
                with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # Run for N algorithms
                    for algo in algorithms:
                        if self._debug:
                            print("Starting thread with %s algorithm" % algo)

                        trainer = Trainer(algo=algo, data=data, target_metric=target_metric, debug=self._debug)
                        futures[executor.submit(trainer.start, max_evals)] = algo

                    # Load results of completed tasks
                    for future in concurrent.futures.as_completed(futures):
                        algo = futures[future]
                        hyperparams, score = future.result()

                        # If no exceptions, then include in tasks, else remove from algorithms list
                        if hyperparams or score:
                            tasks[iteration][algo] = {
                                'hyperparameters': hyperparams,
                                'score': score,
                                'above_baseline': score['loss'] < baseline_loss
                            }
                        else:
                            print('Cannot use algo : %s' % algo)

                            tasks[iteration][algo] = {
                                'above_baseline': False,
                                'exception': True
                            }

                if len(tasks[iteration]) == 1:
                    break
                else:
                    algorithms = strategy.filter_algorithms(tasks[iteration], algorithms)
                    iteration += 1

            best_model = list(tasks[iteration].keys())[0]
            best_params = tasks[iteration][best_model]['hyperparameters']
            best_score = tasks[iteration][best_model]['score']['loss']

            return best_model, best_params, best_score, tasks
        except ValidationError as e:
            """
            Catch validation errors
            """
            print(e.message)
            if self._debug:
                raise
