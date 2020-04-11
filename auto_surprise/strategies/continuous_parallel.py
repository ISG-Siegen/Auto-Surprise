import concurrent.futures

from auto_surprise.constants import (EVALS_MULTIPLIER, MAX_WORKERS)
from auto_surprise.trainer import Trainer
from auto_surprise.strategies.base import StrategyBase

class ContinuousParallel(StrategyBase):
    """
    Executes all alogrithms in parallel until time limit exceeded or max evals reached.
    """

    def evaluate(self):
        """
        Evaluate performance of algorithms
        """

        print("Starting evaluation using strategy : ContinuousParallel")

        tasks = {}
        max_evals = self.max_evals
        futures = {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Run for N algorithms
            for algo in self.algorithms:
                print("Starting thread with %s algorithm" % algo)

                trainer = Trainer(algo=algo, data=self.data, target_metric=self.target_metric, debug=self._debug)
                futures[
                    executor.submit(trainer.start_with_limits, max_evals, time_limit=self.time_limit)
                ] = algo

            # Load results of completed tasks
            for future in concurrent.futures.as_completed(futures):
                algo = futures[future]
                hyperparams, best_trial = future.result()

                if hyperparams or best_trial:
                    tasks[algo] = {
                        'hyperparameters': best_trial['hyperparams'],
                        'score': best_trial,
                        'above_baseline': best_trial['loss'] < self.baseline_loss
                    }
                else:
                    print('Cannot use algo : %s' % algo)

                    tasks[algo] = {
                        'hyperparameters': None,
                        'score': { 'loss': 100 },
                        'above_baseline': False,
                        'exception': True
                    }

        print(tasks)

        best_model = min(tasks.items(), key=(lambda x: x[1]['score']['loss']))[0]
        best_params = tasks[best_model]['hyperparameters']
        best_score = tasks[best_model]['score']['loss']

        return best_model, best_params, best_score, tasks
