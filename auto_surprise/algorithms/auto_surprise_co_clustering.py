from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from surprise import CoClustering
from surprise.model_selection import cross_validate
from auto_surprise.constants import DEFAULT_MAX_EVALS, DEFAULT_TARGET_METRIC, ACCURACY_METRICS, CV_N_JOBS

SPACE = {
    'n_cltr_u': hp.choice('n_cltr_u', range(1,30)),
    'n_cltr_i': hp.choice('n_cltr_i', range(1,30)),
    'n_epochs': hp.choice('n_epochs', range(5,50)),
}

class AutoSurpriseCoClustering(object):
    def __init__(self, cv=5, metric=DEFAULT_TARGET_METRIC, data=None, cv_n_jobs=CV_N_JOBS, debug=False):
        self._cv = cv
        self._metric = metric
        self._data = data
        self._debug = debug
        self._cv_n_jobs = cv_n_jobs

    def _hyperopt(self, params):
        algo = CoClustering(
            n_cltr_u=params['n_cltr_u'],
            n_cltr_i=params['n_epochs'],
            n_epochs=params['n_epochs']
        )
        return cross_validate(algo, self._data, measures=ACCURACY_METRICS, cv=self._cv, n_jobs=self._cv_n_jobs, verbose=self._debug)[self._metric].mean()

    def _objective(self, params):
        loss = self._hyperopt(params)
        return {'loss': loss, 'status': STATUS_OK}

    def best_hyperparams(self, max_evals=DEFAULT_MAX_EVALS):
        trials = Trials()
        best = fmin(self._objective, SPACE, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        return best, trials
