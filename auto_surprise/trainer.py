import sys
import traceback
import logging
from surprise.model_selection import cross_validate

# import algorithms
from auto_surprise.algorithms.auto_surprise_svd import AutoSurpriseSVD
from auto_surprise.algorithms.auto_surprise_svdpp import AutoSurpriseSVDpp
from auto_surprise.algorithms.auto_surprise_nmf import AutoSurpriseNMF
from auto_surprise.algorithms.auto_surprise_knn_baseline import AutoSurpriseKNNBaseline
from auto_surprise.algorithms.auto_surprise_knn_basic import AutoSurpriseKNNBasic
from auto_surprise.algorithms.auto_surprise_knn_with_means import AutoSurpriseKNNWithMeans
from auto_surprise.algorithms.auto_surprise_knn_with_z_score import AutoSurpriseKNNWithZScore
from auto_surprise.algorithms.auto_surprise_slope_one import AutoSurpriseSlopeOne
from auto_surprise.algorithms.auto_surprise_co_clustering import AutoSurpriseCoClustering
from auto_surprise.algorithms.auto_surprise_normal_predictor import AutoSurpriseNormalPredictor
from auto_surprise.algorithms.auto_surprise_baseline_only import AutoSurpriseBaselineOnly

from auto_surprise.constants import ALGORITHM_MAP, DEFAULT_TARGET_METRIC, CV_N_JOBS, DEFAULT_HPO_ALGO
from auto_surprise.context.limits import TimeoutException
from auto_surprise.context.result_logging_manager import ResultLoggingManager
import auto_surprise.context.limits as limits

class Trainer(object):
    def __init__(self, tmp_dir, algo=None, data=None, target_metric=DEFAULT_TARGET_METRIC, hpo_algo=DEFAULT_HPO_ALGO, debug=False):
        """
        Initialize new trainer
        """
        self.__logger = logging.getLogger(__name__)
        self._debug = debug
        self._tmp_dir = tmp_dir
        self._algo_name = algo
        self._algo_class = ALGORITHM_MAP[algo]
        # Dynamically instantiate algorithm
        self.algo = getattr(sys.modules[__name__], self._algo_class)(data=data, metric=target_metric, cv_n_jobs=CV_N_JOBS, hpo_algo=hpo_algo, debug=debug)

    def start(self, max_evals):

        try:
            with ResultLoggingManager(self._tmp_dir, self._algo_class) as result_logger:
                self.algo.set_result_logger(result_logger)

                best, trials = self.algo.best_hyperparams(max_evals)

            best_trial = None
            # Sort best trial based on loss value
            if best:
                best_trial = sorted(trials.results, key=lambda x: x['loss'], reverse=False)[0]
            else:
                best_trial = trials

            return best, best_trial
        except Exception as e:
            self.__logger.ERROR('Exception : ', e)
            self.__logger.ERROR(traceback.format_exc())
            if self._debug:
                raise

            return False, False

    def start_with_limits(self, max_evals, time_limit, tasks):
        # Result logger should not be destroyed by exceptions
        with ResultLoggingManager(self._tmp_dir, self._algo_class) as result_logger:
            try:
                # Although hyperopt.fmin does have a timout parameter, it doesnt seem to work with multiprocessing
                with limits.run_with_enforced_limits(time_limit=time_limit):
                    self.algo.set_result_logger(result_logger)

                    best, best_trial = self.algo.best_hyperparams(max_evals)

                    if self.algo.trials.results:
                        best_trial = sorted(self.algo.trials.results, key=lambda x: x['loss'], reverse=False)[0]

                    tasks[self._algo_name] = {
                        'score': best_trial,
                    }

            except TimeoutException:
                # Handle timeout when enforced cpu time limit is reached
                trials = self.algo.trials

                if trials.results:
                    best_trial = sorted(trials.results, key=lambda x: x['loss'], reverse=False)[0]
                    tasks[self._algo_name] = {
                        'score': best_trial,
                    }
                else:
                    best_trial = False
                    tasks[self._algo_name] = {
                        'score': { 'loss': 100, 'hyperparams': None },
                    }

            except Exception:
                self.__logger.error("Exception for algo {0}".format(self._algo_name))
                self.__logger.error(traceback.format_exc())

                if self._debug:
                    raise

                tasks[self._algo_name] = {
                    'score': { 'loss': 100, 'hyperparams': None },
                    'exception': True
                }
