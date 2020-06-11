import unittest
import pathlib
import utils as test_utils
from auto_surprise.trainer import Trainer

class TrainerTest(unittest.TestCase):
    def setUp(self):
        self.data = test_utils.load_test_dataset()
        self.tmp_path = test_utils.get_tmp_dir()

    def test_start_with_limits(self):
        # TODO: Need to expand on these tests
        trainer = Trainer(self.tmp_path, algo="svd", data=self.data)
        tasks = {}

        # Even with timeout exception, tasks should still be populated
        trainer.start_with_limits(10, 10, tasks)
        self.assertTrue(tasks["svd"])
