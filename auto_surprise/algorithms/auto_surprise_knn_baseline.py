from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from surprise import KNNBaseline
from surprise.model_selection import cross_validate
from auto_surprise.constants import DEFAULT_MAX_EVALS, DEFAULT_TARGET_METRIC

SIMILARITY_OPTIONS_SPACE = {
    'name': hp.choice('name', ['cosine', 'msd', 'pearson', 'pearson_baseline']),
    'user_based': hp.choice('user_based', [False, True]),
    'shrinkage': hp.choice('shrinkage', range(1, 300))
}

SPACE = {
    'k': hp.choice('k', range(1,100)),
    'min_k': hp.choice('min_k', range(1,10)),
    'sim_options': SIMILARITY_OPTIONS_SPACE
}

class AutoSurpriseKNNBaseline(object):
    def __init__(self, cv=5, metric=DEFAULT_TARGET_METRIC, data=None, debug=False):
        self._cv = cv
        self._metric = metric
        self._data = data
        self._debug = debug

    def _hyperopt(self, params):
        print(params)
        algo = KNNBaseline(k=params['k'], min_k=params['min_k'], sim_options=params['sim_options'])
        return cross_validate(algo, self._data, measures=['RMSE', 'MAE'], cv=self._cv, verbose=self._debug)[self._metric].mean()

    def _objective(self, params):
        loss = self._hyperopt(params)
        return {'loss': loss, 'status': STATUS_OK}

    def best_hyperparams(self, max_evals=DEFAULT_MAX_EVALS):
        trials = Trials()
        best = fmin(self._objective, SPACE, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        return best, trials
