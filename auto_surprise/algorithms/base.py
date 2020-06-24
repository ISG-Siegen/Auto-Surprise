from auto_surprise.constants import (
    DEFAULT_TARGET_METRIC,
    CV_N_JOBS,
    DEFAULT_HPO_ALGO,
    DEFAULT_CV_ITERS,
)
from hyperopt import Trials


class AlgorithmBase(object):
    def __init__(
        self,
        cv=DEFAULT_CV_ITERS,
        metric=DEFAULT_TARGET_METRIC,
        data=None,
        cv_n_jobs=CV_N_JOBS,
        hpo_algo=DEFAULT_HPO_ALGO,
        verbose=False,
    ):
        self._cv = cv
        self._metric = metric
        self._data = data
        self.verbose = verbose
        self._cv_n_jobs = cv_n_jobs
        self._hpo_algo = hpo_algo
        self.trials = Trials()

    def set_result_logger(self, result_logger_manager):
        self._result_logger = result_logger_manager
