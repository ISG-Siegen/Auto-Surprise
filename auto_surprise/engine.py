import os
import logging
import pathlib

from auto_surprise.constants import (
    DEFAULT_TARGET_METRIC,
    DEFAULT_MAX_EVALS,
    FULL_ALGO_LIST,
    BASELINE_ALGO,
    EVALS_MULTIPLIER,
    SURPRISE_ALGORITHM_MAP,
    DEFAULT_HPO_ALGO,
)
from auto_surprise.trainer import Trainer
from auto_surprise.exceptions import ValidationError
from auto_surprise.context.backend import BackendContextManager
from auto_surprise.strategies.continuous_parallel import ContinuousParallel
import auto_surprise.validation_util as validation_util


class Engine(object):
    def __init__(self, verbose=True, algorithms=FULL_ALGO_LIST):
        """
        Initialize new engine
        """
        self.__logger = logging.getLogger(__name__)
        self.verbose = verbose
        self.algorithms = algorithms
        self._current_path = pathlib.Path().absolute()

    def train(
        self,
        target_metric=DEFAULT_TARGET_METRIC,
        data=None,
        max_evals=DEFAULT_MAX_EVALS,
        cpu_time_limit=None,
        hpo_algo=DEFAULT_HPO_ALGO,
    ):
        """
        Train and find most optimal model and hyperparameters
        """

        try:
            # Validations
            validation_util.validate_target_metric(target_metric)
            validation_util.validate_dataset(data)
            validation_util.validate_max_evals(max_evals)
            validation_util.validate_cpu_time_limit(cpu_time_limit)
        except ValidationError as err:
            """
            Catch validation errors. Auto-Surprise cannot run with these exceptions
            """
            self.__logger.critical(err.message)
            raise err

        # Determine baseline value from random normal predictor
        with BackendContextManager(self._current_path) as tmp_dir:
            if self.verbose:
                print("Available CPUs: {0}".format(os.cpu_count()))

            # Calculate baseline first. This can be used to early stop training of algorithms if results are not optimal
            baseline_trainer = Trainer(
                tmp_dir,
                algo=BASELINE_ALGO,
                data=data,
                target_metric=target_metric,
                verbose=self.verbose,
            )
            baseline_loss = baseline_trainer.start(1)[1]["loss"]
            if self.verbose:
                print("Baseline loss : {0}".format(baseline_loss))

            # Initialize the strategy to be used to optimize. Currently only one strategy implemented.
            strategy = ContinuousParallel(
                self.algorithms,
                data,
                target_metric,
                baseline_loss,
                tmp_dir,
                time_limit=cpu_time_limit,
                max_evals=max_evals,
                hpo_algo=hpo_algo,
                verbose=self.verbose,
            )

            best_algo, best_params, best_score, tasks = strategy.evaluate()

        if self.verbose:
            print("----Done!----")
            print("Best algorithm: {0}".format(best_algo))
            print("Best hyperparameters: {0}".format(best_params))

        return best_algo, best_params, best_score, tasks

    def build_model(self, algo_name, params):
        algo = SURPRISE_ALGORITHM_MAP[algo_name]
        if params:
            return algo(**params)
        else:
            return algo()
