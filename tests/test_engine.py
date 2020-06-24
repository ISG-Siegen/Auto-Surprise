import unittest
import utils as test_utils
from auto_surprise.engine import Engine
from auto_surprise.constants import FULL_ALGO_LIST

class TestEngine(unittest.TestCase):
    
    def setUp(self):
        self.data = test_utils.load_test_dataset()
        self.engine = Engine()

    def test_trainer(self):

        with self.subTest("Sanity - everything just works"):
            best_algo, best_params, best_score, tasks=self.engine.train(data=self.data, cpu_time_limit=60, max_evals=20)
            self.assertTrue(best_algo)
            self.assertTrue(best_score)
            self.assertTrue(tasks)

            self.assertCountEqual(list(tasks.keys()), FULL_ALGO_LIST)
