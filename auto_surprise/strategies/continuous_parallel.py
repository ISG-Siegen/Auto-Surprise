import logging
import multiprocessing
from auto_surprise.constants import EVALS_MULTIPLIER, MAX_WORKERS
from auto_surprise.trainer import Trainer
from auto_surprise.strategies.base import StrategyBase


class ContinuousParallel(StrategyBase):
    """
    Executes all alogrithms in parallel until time limit exceeded or max evals reached.
    """

    def __init__(self, *args, **kwargs):
        self.__logger = logging.getLogger(__name__)
        super().__init__(*args, **kwargs)

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
                if self.verbose:
                    print("Starting process with %s algorithm" % algo)

                trainer = Trainer(
                    self.tmp_dir,
                    algo=algo,
                    data=self.data,
                    target_metric=self.target_metric,
                    hpo_algo=self.hpo_algo,
                    verbose=self.verbose,
                )
                p = multiprocessing.Process(
                    target=trainer.start_with_limits,
                    args=(max_evals, self.time_limit, tasks),
                )
                processes.append(p)
                p.start()

            # Wait for processes to complete
            for process in processes:
                process.join()

            passed_jobs = {k:v for k,v in tasks.items() if v["score"]["loss"]}
            best_algo = min(passed_jobs.items(), key=(lambda x: x[1]["score"]["loss"]))[0]
            best_params = passed_jobs[best_algo]["score"]["hyperparams"]
            best_score = passed_jobs[best_algo]["score"]["loss"]

            return best_algo, best_params, best_score, tasks.copy()
