import os
import pathlib

from auto_surprise.constants import (
    DEFAULT_TARGET_METRIC, DEFAULT_MAX_EVALS,
    FULL_ALGO_LIST, QUICK_COMPUTE_ALGO_LIST,
    BASELINE_ALGO, EVALS_MULTIPLIER,
    SURPRISE_ALGORITHM_MAP, DEFAULT_HPO_ALGO
)
from auto_surprise.trainer import Trainer
from auto_surprise.exceptions import ValidationError
from auto_surprise.context.backend import BackendContextManager
from auto_surprise.strategies.basic_reduction import BasicReduction
from auto_surprise.strategies.continuous_parallel import ContinuousParallel
import auto_surprise.validation_util as validation_util

class Engine(object):
    def __init__(self, debug=False):
        """
        Initialize new engine
        """
        self._debug = debug
        self._current_path = pathlib.Path().absolute()

    def train(
        self,
        target_metric=DEFAULT_TARGET_METRIC,
        data=None,
        max_evals=DEFAULT_MAX_EVALS,
        quick_compute=False,
        cpu_time_limit=None,
        hpo_algo=DEFAULT_HPO_ALGO,
        strategy="continuos_parallel"
    ):
        """
        Train and find most optimal model and hyperparameters
        """

        try:
            # Validations
            validation_util.validate_target_metric(target_metric)
            validation_util.validate_dataset(data)
            validation_util.validate_max_evals(max_evals)
        except ValidationError as e:
            """
            Catch validation errors
            """
            print(e.message)
            if self._debug:
                raise

        # Determine baseline value from random normal predictor
        with BackendContextManager(self._current_path) as tmp_dir:
            print("Available CPUs: {0}".format(os.cpu_count()))
            baseline_trainer = Trainer(tmp_dir, algo=BASELINE_ALGO, data=data, target_metric=target_metric, debug=self._debug)
            baseline_loss = baseline_trainer.start(1)[1]['loss']
            print("Baseline loss : {0}".format(baseline_loss))

            algorithms = QUICK_COMPUTE_ALGO_LIST if quick_compute else FULL_ALGO_LIST

            # Select the strategy
            if strategy == 'continuos_parallel':
                strategy = ContinuousParallel(
                    algorithms,
                    data,
                    target_metric,
                    baseline_loss,
                    tmp_dir,
                    max_evals=max_evals,
                    time_limit=cpu_time_limit,
                    hpo_algo=hpo_algo,
                    debug=self._debug
                )
            else:
                strategy = BasicReduction(
                    algorithms,
                    data,
                    target_metric,
                    baseline_loss,
                    tmp_dir,
                    hpo_algo=hpo_algo,
                    debug=self._debug
                )

            best_model, best_params, best_score, tasks = strategy.evaluate()

        return best_model, best_params, best_score, tasks

    def build_model(self, algo_name, params):
        algo = SURPRISE_ALGORITHM_MAP[algo_name]
        if (params):
            return algo(**params)
        else:
            return algo()
