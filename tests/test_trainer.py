import unittest
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
            self.assertListEqual(list(tasks["svd"].keys()), ["loss", "status", "hyperparams", "exception", "trials"])

        with self.subTest(msg="With timeout exception, trainer should still populate tasks with exception tag false"):
            tasks = {}
            trainer.start_with_limits(100, 10, tasks)
            self.assertTrue(tasks["svd"])
            self.assertEqual(tasks["svd"]["exception"], False)

        with self.subTest(msg="With timeout exception, but the job was not evaluated even once"):
            tasks = {}
            trainer = Trainer(self.tmp_path, algo="knn_baseline", data=self.data)           
            trainer.start_with_limits(100, 1, tasks)
            self.assertTrue(tasks["knn_baseline"])
            self.assertEqual(tasks["knn_baseline"]["exception"], False)
            # This is a bit hard to assert, since sometimes it can evaluate in 1 sec. Need to revisit this test later
            # self.assertIsNone(tasks["knn_baseline"]["loss"])
