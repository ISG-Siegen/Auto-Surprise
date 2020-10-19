import os
import logging
import pathlib
import numpy
from rich.console import Console
from rich.table import Column, Table

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
from auto_surprise.strategies.smbo import SMBO
from auto_surprise.__version__ import __version__
import auto_surprise.validation_util as validation_util


class Engine(object):
    def __init__(
        self, verbose=True, algorithms=FULL_ALGO_LIST, random_state=numpy.random
    ):
        """
        Initialize new engine
        """
        self.__logger = logging.getLogger(__name__)
        self.verbose = verbose
        self.algorithms = algorithms
        self.random_state = random_state
        self._current_path = pathlib.Path().absolute()

        if self.verbose:
            self.console = Console()

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

        if self.verbose:
            self.console.print("auto_surprise {0}".format(__version__))

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
                self.console.print("Available CPUs: {0}".format(os.cpu_count()))

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
                self.console.print("Baseline loss: {0}".format(baseline_loss))

            # Initialize the strategy to be used to optimize. Currently only one strategy implemented.
            strategy = SMBO(
                self.algorithms,
                data,
                target_metric,
                baseline_loss,
                tmp_dir,
                time_limit=cpu_time_limit,
                max_evals=max_evals,
                hpo_algo=hpo_algo,
                verbose=self.verbose,
                random_state=self.random_state,
            )

            best_algo, best_params, best_score, tasks = strategy.evaluate()

        if self.verbose:
            self.console.print("----Done!----")
            self.console.print("Best algorithm: {0}".format(best_algo))
            self.console.print("Best hyperparameters: {0}".format(best_params))
            self.print_results_table(tasks)

        return best_algo, best_params, best_score, tasks

    def build_model(self, algo_name, params):
        algo = SURPRISE_ALGORITHM_MAP[algo_name]
        if params:
            return algo(**params)
        else:
            return algo()

    def print_results_table(self, tasks):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Algorithm")
        table.add_column("Hyperparameters")
        table.add_column("Loss", justify="right")

        for key, val in tasks.items():
            table.add_row(key, str(val["hyperparams"]), str(val["loss"]))

        self.console.print(table)
