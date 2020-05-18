import concurrent.futures

from auto_surprise.constants import (EVALS_MULTIPLIER, MAX_WORKERS)
from auto_surprise.trainer import Trainer
from auto_surprise.strategies.base import StrategyBase

class BasicReduction(StrategyBase):
    """
    A basic strategy for comparison of algorithms
    """

    def evaluate(self):
        """
        Evaluate performance of algorithms
        """

        print("Starting evaluation using strategy : BasicReduction")

        tasks = {}
        iteration = 0
        while True:
            print("Iteration: %i" % iteration)

            max_evals = EVALS_MULTIPLIER * (iteration + 1)
            tasks[iteration] = {}
            futures = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Run for N algorithms
                for algo in self.algorithms:
                    print("Starting thread with %s algorithm" % algo)

                    trainer = Trainer(algo=algo, data=self.data, target_metric=self.target_metric, hpo_algo=self.hpo_algo, debug=self._debug)
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
                            'above_baseline': score['loss'] < self.baseline_loss
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
                self.__filter_algorithms(tasks[iteration])
                iteration += 1

        best_model = list(tasks[iteration].keys())[0]
        best_params = tasks[iteration][best_model]['hyperparameters']
        best_score = tasks[iteration][best_model]['score']['loss']

        return best_model, best_params, best_score, tasks

    def __filter_algorithms(self, tasks):
        """
        Rank N algorithms and take the top N/2 algorithms which performed
        better than baseline result for the next iteration
        """

        filtered_algorithms = dict(filter(lambda algo: algo[1]['above_baseline'], tasks.items()))
        algorithms_ranking = [i[0] for i in sorted(filtered_algorithms.items(), key=lambda x: x[1]['score']['loss'], reverse=False)]
        algorithms_count = round(len(self.algorithms) / 2)
        algorithms = algorithms_ranking[0:algorithms_count]

        return algorithms
