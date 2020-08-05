import sys
import traceback
import logging

from surprise.model_selection import cross_validate
from auto_surprise.algorithms.base import AlgorithmBase
from auto_surprise.constants import (
    DEFAULT_TARGET_METRIC,
    CV_N_JOBS,
    DEFAULT_HPO_ALGO,
)
from auto_surprise.context.limits import TimeoutException
from auto_surprise.context.result_logging_manager import ResultLoggingManager
import auto_surprise.context.limits as limits


class Trainer(object):
    def __init__(
        self,
        tmp_dir,
        algo=None,
        data=None,
        target_metric=DEFAULT_TARGET_METRIC,
        hpo_algo=DEFAULT_HPO_ALGO,
        verbose=False,
        random_state=None,
        baseline_loss=None,
    ):
        """
        Initialize new trainer
        """
        self.__logger = logging.getLogger(__name__)
        self.verbose = verbose
        self.tmp_dir = tmp_dir
        self.algo_name = algo

        # Dynamically instantiate algorithm
        self.algo_base = AlgorithmBase(
            data=data,
            algo_name=self.algo_name,
            metric=target_metric,
            cv_n_jobs=CV_N_JOBS,
            hpo_algo=hpo_algo,
            verbose=verbose,
            random_state=random_state,
            baseline_loss=baseline_loss,
        )

    def start(self, max_evals):
        """
        Start with no limits
        """
        try:
            with ResultLoggingManager(self.tmp_dir, self.algo_name) as result_logger:
                self.algo_base.set_result_logger(result_logger)

                best, trials = self.algo_base.best_hyperparams(max_evals)

                best_trial = None
                # Sort best trial based on loss value
                if best:
                    best_trial = sorted(
                        trials.results, key=lambda x: x["loss"], reverse=False
                    )[0]
                else:
                    best_trial = trials

                return best, best_trial
        except Exception as e:
            self.__logger.error("Exception : ", e)
            self.__logger.error(traceback.format_exc())

            return False, False

    def start_with_limits(self, max_evals, time_limit, tasks):
        """
        Start the trainer with a time limit
        """
        with ResultLoggingManager(self.tmp_dir, self.algo_name) as result_logger:
            try:
                # Although hyperopt.fmin does have a timout parameter, it doesnt seem to work with multiprocessing
                with limits.run_with_enforced_limits(time_limit=time_limit):
                    self.algo_base.set_result_logger(result_logger)

                    _, best_trial = self.algo_base.best_hyperparams(max_evals)

                    if self.algo_base.trials.results:
                        best_trial = sorted(
                            self.algo_base.trials.results,
                            key=lambda x: x["loss"],
                            reverse=False,
                        )[0]

                    tasks[self.algo_name] = {
                        **best_trial,
                        "exception": False,
                        "trials": self.algo_base.trials,
                    }

            except TimeoutException:
                # Handle timeout when enforced cpu time limit is reached
                trials = self.algo_base.trials

                if trials.results:
                    best_trial = sorted(
                        trials.results, key=lambda x: x["loss"], reverse=False
                    )[0]
                    # A timeout exception is not considered an algorithm exception
                    tasks[self.algo_name] = {
                        **best_trial,
                        "exception": False,
                        "trials": trials,
                    }
                else:
                    # When no trials were completed before the job timed out
                    tasks[self.algo_name] = {
                        "loss": None,
                        "hyperparams": None,
                        "exception": False,
                        "trials": trials,
                    }

            except Exception:
                self.__logger.error("Exception for algo {0}".format(self.algo_name))
                self.__logger.error(traceback.format_exc())

                tasks[self.algo_name] = {
                    "loss": None,
                    "hyperparams": None,
                    "exception": True,
                    "trials": self.algo_base.trials,
                }
