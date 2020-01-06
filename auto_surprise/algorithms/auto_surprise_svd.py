from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from surprise import SVD
from surprise.model_selection import cross_validate
from auto_surprise.constants import DEFAULT_MAX_EVALS, DEFAULT_TARGET_METRIC

SPACE = {
    'n_factors': hp.choice('n_factors', range(1, 100)),
    'n_epochs': hp.choice('n_epochs', range(1, 20)),
    'lr_all': hp.uniform('lr_all', 0.001, 0.3),
    'reg_all': hp.uniform('reg_all', 0.001, 0.3),
}

class AutoSurpriseSVD(object):
    def __init__(self, cv=5, metric=DEFAULT_TARGET_METRIC, data=None, debug=False):
        self._cv = cv
        self._metric = metric
        self._data = data
        self._debug = debug

    def _hyperopt(self, params):
        print(params)
        algo = SVD(n_factors=params['n_factors'], n_epochs=params['n_epochs'], lr_all=params['lr_all'], reg_all=params['reg_all'])
        return cross_validate(algo, self._data, measures=['RMSE', 'MAE'], cv=self._cv, verbose=self._debug)[self._metric].mean()

    def _objective(self, params):
        loss = self._hyperopt(params)
        return {'loss': loss, 'status': STATUS_OK}

    def best_hyperparams(self, max_evals=DEFAULT_MAX_EVALS):
        trials = Trials()
        best = fmin(self._objective, SPACE, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        return best, trials
