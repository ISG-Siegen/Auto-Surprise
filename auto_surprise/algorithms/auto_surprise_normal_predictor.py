from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from surprise import NormalPredictor
from surprise.model_selection import cross_validate
from auto_surprise.constants import DEFAULT_MAX_EVALS, DEFAULT_TARGET_METRIC, ACCURACY_METRICS, CV_N_JOBS

"""
Only to be used for calculating baselines to evaluate against other algorithms
"""
class AutoSurpriseNormalPredictor(object):
    def __init__(self, cv=5, metric=DEFAULT_TARGET_METRIC, data=None, cv_n_jobs=CV_N_JOBS, debug=False):
        self._cv = cv
        self._metric = metric
        self._data = data
        self._debug = debug
        self._cv_n_jobs = cv_n_jobs
        
    def _hyperopt(self):
        algo = NormalPredictor()
        return cross_validate(algo, self._data, measures=ACCURACY_METRICS, cv=self._cv, n_jobs=self._cv_n_jobs, verbose=self._debug)[self._metric].mean()

    def _objective(self):
        loss = self._hyperopt()
        return { 'loss': loss, 'status': STATUS_OK }

    def best_hyperparams(self, max_evals=DEFAULT_MAX_EVALS):
        best = self._objective()
        return None, best
