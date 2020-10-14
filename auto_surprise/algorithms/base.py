from hyperopt import Trials, fmin, STATUS_OK
from surprise import (
    SVD,
    SVDpp,
    NMF,
    KNNBaseline,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    SlopeOne,
    CoClustering,
    NormalPredictor,
    BaselineOnly,
)
from surprise.model_selection import cross_validate
from auto_surprise.constants import (
    DEFAULT_TARGET_METRIC,
    CV_N_JOBS,
    DEFAULT_HPO_ALGO,
    DEFAULT_CV_ITERS,
    SURPRISE_ALGORITHM_MAP,
    DEFAULT_MAX_EVALS,
    ACCURACY_METRICS,
)
from auto_surprise.algorithms.spaces import HPO_SPACE_MAP


class AlgorithmBase(object):
    def __init__(
        self,
        algo_name=None,
        cv=DEFAULT_CV_ITERS,
        metric=DEFAULT_TARGET_METRIC,
        data=None,
        cv_n_jobs=CV_N_JOBS,
        hpo_algo=DEFAULT_HPO_ALGO,
        verbose=False,
        random_state=None,
        baseline_loss=None
    ):
        self.algo = SURPRISE_ALGORITHM_MAP[algo_name]
        self.cv = cv
        self.metric = metric
        self.data = data
        self.verbose = verbose
        self.cv_n_jobs = cv_n_jobs
        self.hpo_algo = hpo_algo
        self.trials = Trials()
        self.space = HPO_SPACE_MAP[algo_name]
        self.random_state = random_state
        self.baseline_loss = baseline_loss
        self._result_logger = None
    
    def set_result_logger(self, result_logger_manager):
        self._result_logger = result_logger_manager

    def objective(self, params):
        if params:
            algo = self.algo(**params)
        else:
            algo = self.algo()

        loss = cross_validate(
            algo,
            self.data,
            measures=ACCURACY_METRICS,
            cv=self.cv,
            n_jobs=self.cv_n_jobs,
            verbose=self.verbose,
        )[self.metric].mean()

        if self._result_logger:
            self._result_logger.append_results(loss, params)

        return {"loss": loss, "status": STATUS_OK, "hyperparams": params}

    def early_baseline_loss_stop(self, trials, *stop_args):
        best_trial = trials.best_trial
        if len(trials) == 10 and best_trial and self.baseline_loss < best_trial["result"]["loss"]:
            return True, { "failed_baseline_check": True }
        else:
            return False, { "failed_baseline_check": False }

    def best_hyperparams(self, max_evals):
        if self.space:
            fmin_args = {
                "fn": self.objective,
                "space": self.space,
                "algo": self.hpo_algo,
                "max_evals": max_evals,
                "trials": self.trials,
                "verbose": self.verbose,
                "rstate": self.random_state,
            }

            if self.baseline_loss:
                fmin_args["early_stop_fn"] = self.early_baseline_loss_stop
            
            best = fmin(**fmin_args)

            return best, self.trials
        else:
            best = self.objective(None)
            return None, best
