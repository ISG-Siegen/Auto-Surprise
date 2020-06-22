from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from surprise import SlopeOne
from surprise.model_selection import cross_validate
from auto_surprise.constants import DEFAULT_MAX_EVALS, ACCURACY_METRICS
from auto_surprise.algorithms.base import AlgorithmBase


class AutoSurpriseSlopeOne(AlgorithmBase):
    """
    Wrapper for surprise.prediction_algorithms.slope_one.SlopeOne algorithm
    SlopeOne does not require any hyperparameter tuning
    """

    def _hyperopt(self):
        algo = SlopeOne()
        return cross_validate(
            algo,
            self._data,
            measures=ACCURACY_METRICS,
            cv=self._cv,
            n_jobs=self._cv_n_jobs,
            verbose=self.verbose,
        )[self._metric].mean()

    def _objective(self):
        loss = self._hyperopt()
        self._result_logger.append_results(loss, None)

        return {"loss": loss, "status": STATUS_OK, "hyperparams": None}

    def best_hyperparams(self, max_evals):
        best = self._objective()
        # No hyperparameters used
        return None, best
