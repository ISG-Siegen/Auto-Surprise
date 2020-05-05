import time
import datetime
import pathlib

class ResultLoggingManager():
    def __init__(self, tmp_dir, algo_name):
        self._start_time = time.time()
        self._log_file_path = tmp_dir / "{0}_results.csv".format(algo_name)

    def __enter__(self):
        self._log_file_writer = open(self._log_file_path, "w", buffering=1)
        # Set headers
        self._log_file_writer.write("time,loss\n")
        return self

    def __exit__(self, *exc):
        self._log_file_writer.close()

    def append_results(self, loss):
        execution_time = str(datetime.timedelta(seconds=int(time.time() - self._start_time)))
        self._log_file_writer.write("{0},{1}\n".format(execution_time, loss))
