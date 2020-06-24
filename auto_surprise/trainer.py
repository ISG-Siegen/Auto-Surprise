import sys
import traceback
import logging

from surprise.model_selection import cross_validate
from auto_surprise.algorithms import (
    AutoSurpriseBaselineOnly,
    AutoSurpriseCoClustering,
    AutoSurpriseKNNBaseline,
    AutoSurpriseKNNBasic,
    AutoSurpriseKNNWithMeans,
    AutoSurpriseKNNWithZScore,
    AutoSurpriseNMF,
    AutoSurpriseSlopeOne,
    AutoSurpriseSVD,
    AutoSurpriseSVDpp,
    AutoSurpriseNormalPredictor,
)
from auto_surprise.constants import (
    ALGORITHM_MAP,
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
    ):
        """
        Initialize new trainer
        """
        self.__logger = logging.getLogger(__name__)
        self.verbose = verbose
        self._tmp_dir = tmp_dir
        self._algo_name = algo
        self._algo_class = ALGORITHM_MAP[algo]

        # Dynamically instantiate algorithm
        self.algo = getattr(sys.modules[__name__], self._algo_class)(
            data=data,
            metric=target_metric,
            cv_n_jobs=CV_N_JOBS,
            hpo_algo=hpo_algo,
            verbose=verbose,
        )

    def start(self, max_evals):

        try:
            with ResultLoggingManager(self._tmp_dir, self._algo_class) as result_logger:
                self.algo.set_result_logger(result_logger)

                best, trials = self.algo.best_hyperparams(max_evals)

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
            self.__logger.ERROR("Exception : ", e)
            self.__logger.ERROR(traceback.format_exc())

            return False, False

    def start_with_limits(self, max_evals, time_limit, tasks):
        """
        Start the trainer with a time limit
        """
        with ResultLoggingManager(self._tmp_dir, self._algo_class) as result_logger:
            try:
                # Although hyperopt.fmin does have a timout parameter, it doesnt seem to work with multiprocessing
                with limits.run_with_enforced_limits(time_limit=time_limit):
                    self.algo.set_result_logger(result_logger)

                    _, best_trial = self.algo.best_hyperparams(max_evals)

                    if self.algo.trials.results:
                        best_trial = sorted(
                            self.algo.trials.results,
                            key=lambda x: x["loss"],
                            reverse=False,
                        )[0]

                    tasks[self._algo_name] = {
                        **best_trial,
                        "exception": False
                    }

            except TimeoutException:
                # Handle timeout when enforced cpu time limit is reached
                trials = self.algo.trials

                if trials.results:
                    best_trial = sorted(
                        trials.results, key=lambda x: x["loss"], reverse=False
                    )[0]
                    # A timeout exception is not considered an algorithm exception
                    tasks[self._algo_name] = {
                        **best_trial,
                        "exception": False
                    }
                else:
                    # When no trials were completed before the job timed out
                    tasks[self._algo_name] = {
                        "loss": None, 
                        "hyperparams": None,
                        "exception": False,
                    }

            except Exception:
                self.__logger.error("Exception for algo {0}".format(self._algo_name))
                self.__logger.error(traceback.format_exc())

                tasks[self._algo_name] = {
                    "loss": None, 
                    "hyperparams": None,
                    "exception": True,
                }
