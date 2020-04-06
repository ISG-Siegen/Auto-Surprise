from datetime import datetime
import pathlib

class BackendContextManager():
    def __init__(self, current_path):
        date_time = datetime.now.strftime("%m%d%YH%M%S")
        self._temporary_directory = current_path / 'tmp' / date_time

    def __enter__(self):
        self.create_directories()

    def __exit__(self):
        print(
            "Stored temporary files in {}. This can be removed",
            self._temporary_directory.as_posix()
        )

    def create_directories(self):
        self._temporary_directory.mkdir()
