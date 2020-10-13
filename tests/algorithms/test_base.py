import unittest
import tests.utils as test_utils
from auto_surprise.algorithms.base import AlgorithmBase

class TestBase(unittest.TestCase):
    def setUp(self):
        self.data = test_utils.load_test_dataset()
        self.tmp_path = test_utils.get_tmp_dir()

    def test_best_hyperparameter(self):

        with self.subTest(msg="Sanity Test"):
            algo_base = AlgorithmBase(
                data=self.data,
                algo_name="knn_basic",
                metric="test_rmse",
                baseline_loss=None,
            )

            best, trials = algo_base.best_hyperparams(5)
            self.assertTrue(best)
            self.assertTrue(trials)

        with self.subTest(msg="When algorithm does not support hyperparameter optimization"):
            algo_base = AlgorithmBase(
                data=self.data,
                algo_name="slope_one",
                metric="test_rmse",
                baseline_loss=None,
            )

            best, trials = algo_base.best_hyperparams(5)
            self.assertFalse(best)
            self.assertTrue(trials)

        with self.subTest(msg="Test when baseline_loss is specified"):
            # Baseline loss is validated after 10 evaluations, so max evals should be set to >10
            # and expected trials lenght is 10
            algo_base= AlgorithmBase(
                data=self.data,
                algo_name="knn_basic",
                metric="test_rmse",
                baseline_loss=0.00001,
            )

            best, trials = algo_base.best_hyperparams(20)
            self.assertTrue(best)
            self.assertTrue(trials)
            self.assertEqual(len(trials), 10)
