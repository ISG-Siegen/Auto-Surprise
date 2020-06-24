import unittest
import pathlib
import utils as test_utils
from auto_surprise.trainer import Trainer

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.data = test_utils.load_test_dataset()
        self.tmp_path = test_utils.get_tmp_dir()

    def test_start_with_limits(self):
        trainer = Trainer(self.tmp_path, algo="svd", data=self.data)

        with self.subTest(msg="With no exceptions, trainer should populate tasks"):
            tasks = {}
            trainer.start_with_limits(2, 10, tasks)
            self.assertTrue(tasks["svd"])
            self.assertListEqual(list(tasks["svd"].keys()), ["loss", "status", "hyperparams", "exception"])

        with self.subTest(msg="With timeout exception, trainer should still populate tasks with exception tag false"):
            tasks = {}
            trainer.start_with_limits(100, 10, tasks)
            self.assertTrue(tasks["svd"])
            self.assertEqual(tasks["svd"]["exception"], False)
