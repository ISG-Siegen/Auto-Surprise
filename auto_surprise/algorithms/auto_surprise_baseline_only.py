from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from surprise import BaselineOnly
from surprise.model_selection import cross_validate
from auto_surprise.constants import DEFAULT_MAX_EVALS, ACCURACY_METRICS
from auto_surprise.algorithms.base import AlgorithmBase

class AutoSurpriseBaselineOnly(AlgorithmBase):
    def _hyperopt(self):
        algo = BaselineOnly()
        return cross_validate(algo, self._data, measures=ACCURACY_METRICS, cv=self._cv, n_jobs=self._cv_n_jobs, verbose=self._debug)[self._metric].mean()

    def _objective(self):
        loss = self._hyperopt()
        self._result_logger.append_results(loss)

        return {
            'loss': loss,
            'status': STATUS_OK,
            'hyperparams': None
        }

    def best_hyperparams(self, max_evals):
        best = self._objective()
        return None, best
