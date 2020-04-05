
class StrategyBase():
    def __init__(
        self,
        algorithms,
        data,
        target_metric,
        baseline_loss,
        debug=False,
        time_limit=None
    ):
        self.algorithms = algorithms
        self.data = data
        self.target_metric = target_metric
        self.baseline_loss = baseline_loss
        self.time_limit = time_limit
        
        self._debug = debug
