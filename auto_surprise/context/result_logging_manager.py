import time
import datetime
import pathlib
import logging

class ResultLoggingManager:
    """
    Log results of each algorithm
    """

    def __init__(self, tmp_dir, algo_name):
        self._start_time = time.time()
        self._log_file_path = tmp_dir / "{0}_results.log".format(algo_name)

        self.logger = logging.getLogger("{0}_{1}".format(__name__, algo_name))
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self._log_file_path)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def __enter__(self):
        self.logger.info("Started hyperarameter optimization")
        return self

    def __exit__(self, *exc):
        self.logger.info("Done!")

    def append_results(self, loss, hyperparams):
        execution_time = str(
            datetime.timedelta(seconds=int(time.time() - self._start_time))
        )
        self.logger.info(
            "Execution time: {0} Loss: {1} Hyperparameters: {2}".format(
                execution_time, loss, hyperparams
            )
        )