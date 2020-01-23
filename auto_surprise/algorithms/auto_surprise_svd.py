from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from surprise import SVD
from surprise.model_selection import cross_validate
from auto_surprise.constants import DEFAULT_MAX_EVALS, DEFAULT_TARGET_METRIC, ACCURACY_METRICS, CV_N_JOBS

SPACE = {
    'n_factors': hp.choice('n_factors', range(1, 100)),
    'n_epochs': hp.choice('n_epochs', range(1, 20)),
    'lr_all': hp.uniform('lr_all', 0.001, 0.3),
    'reg_all': hp.uniform('reg_all', 0.001, 0.3),
}

class AutoSurpriseSVD(object):
    def __init__(self, cv=5, metric=DEFAULT_TARGET_METRIC, data=None, cv_n_jobs=CV_N_JOBS, debug=False):
        self._cv = cv
        self._metric = metric
        self._data = data
        self._debug = debug
        self._cv_n_jobs = cv_n_jobs

    def _hyperopt(self, params):
        print(params)
        algo = SVD(n_factors=params['n_factors'], n_epochs=params['n_epochs'], lr_all=params['lr_all'], reg_all=params['reg_all'])
        return cross_validate(algo, self._data, measures=ACCURACY_METRICS, cv=self._cv, n_jobs=self._cv_n_jobs, verbose=self._debug)[self._metric].mean()

    def _objective(self, params):
        loss = self._hyperopt(params)
        return {'loss': loss, 'status': STATUS_OK}

    def best_hyperparams(self, max_evals=DEFAULT_MAX_EVALS):
        trials = Trials()
        best = fmin(self._objective, SPACE, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        return best, trials
