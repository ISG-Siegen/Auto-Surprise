from auto_surprise.constants import DEFAULT_MAX_EVALS

class StrategyBase():
    def __init__(
        self,
        algorithms,
        data,
        target_metric,
        baseline_loss,
        temporary_directory,
        time_limit=None,
        max_evals=DEFAULT_MAX_EVALS,
        debug=False,
    ):
        self.algorithms = algorithms
        self.data = data
        self.target_metric = target_metric
        self.baseline_loss = baseline_loss
        self.time_limit = time_limit
        self.tmp_dir = temporary_directory

        self.max_evals = max_evals
        self._debug = debug
