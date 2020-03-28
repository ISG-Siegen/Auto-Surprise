from auto_surprise.constants import DEFAULT_TARGET_METRIC, CV_N_JOBS
from hyperopt import Trials

class AlgorithmBase(object):
    def __init__(self, cv=5, metric=DEFAULT_TARGET_METRIC, data=None, cv_n_jobs=CV_N_JOBS, debug=False):
        self._cv = cv
        self._metric = metric
        self._data = data
        self._debug = debug
        self._cv_n_jobs = cv_n_jobs
        self.trials = Trials()
