import logging
import multiprocessing
from auto_surprise.constants import (EVALS_MULTIPLIER, MAX_WORKERS)
from auto_surprise.trainer import Trainer
from auto_surprise.strategies.base import StrategyBase

class ContinuousParallel(StrategyBase):
    """
    Executes all alogrithms in parallel until time limit exceeded or max evals reached.
    """
    def __init__(self, *args):
        self.__logger = logging.getLogger(__name__)
        super().__init__(*args)

    def evaluate(self):
        """
        Evaluate performance of algorithms
        """

        self.__logger.info("Starting evaluation using strategy : ContinuousParallel")

        max_evals = self.max_evals
        processes = []

        # Run for N algorithms
        with multiprocessing.Manager() as mp_manager:

            tasks = mp_manager.dict()

            for algo in self.algorithms:
                self.__logger.debug("Starting process with %s algorithm" % algo)

                trainer = Trainer(self.tmp_dir, algo=algo, data=self.data, target_metric=self.target_metric, hpo_algo=self.hpo_algo, debug=self._debug)
                p = multiprocessing.Process(target=trainer.start_with_limits, args=(max_evals, self.time_limit, tasks))
                processes.append(p)
                p.start()

            # Wait for processes to complete
            for process in processes:
                process.join()

            best_model = min(tasks.items(), key=(lambda x: x[1]['score']['loss']))[0]
            best_params = tasks[best_model]['score']['hyperparams']
            best_score = tasks[best_model]['score']['loss']

            return best_model, best_params, best_score, tasks.copy()
