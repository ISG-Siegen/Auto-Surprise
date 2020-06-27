import unittest
import auto_surprise.validation_util as validation_util
from auto_surprise.exceptions import ValidationError

import utils as test_utils

class TestValidationUtil(unittest.TestCase):
    def test_validate_target_metric(self):
        """
        Test that a valid target metric is selected
        If the metric is not valid, an exception should be thrown
        """

        valid_metric = "test_rmse"
        self.assertTrue(validation_util.validate_target_metric(valid_metric))

        invalid_metric = "recall"
        self.assertRaises(ValidationError, validation_util.validate_target_metric, invalid_metric)

    def test_validate_dataset(self):
        """
        Validate that the given dataset is in the required format
        """

        self.assertRaises(ValidationError, validation_util.validate_dataset, None)
        self.assertRaises(ValidationError, validation_util.validate_dataset, True)
        valid_dataset = test_utils.load_test_dataset()
        self.assertTrue(validation_util.validate_dataset(valid_dataset))

    def test_validate_max_evals(self):
        """
        Validate that max evals is an integer greater than 0
        """

        self.assertRaises(ValidationError, validation_util.validate_max_evals, "3.14")
        self.assertTrue(validation_util.validate_max_evals(10))

    def test_validate_cpu_time_limit(self):
        """
        Validate that max evals is an integer greater than 0
        """

        self.assertRaises(ValidationError, validation_util.validate_cpu_time_limit, "0.451")
        self.assertTrue(validation_util.validate_cpu_time_limit(10))
