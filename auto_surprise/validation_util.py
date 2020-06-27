from auto_surprise.constants import AVAILABLE_METRICS
from auto_surprise.exceptions import ValidationError
from surprise.dataset import DatasetAutoFolds


def validate_target_metric(metric):
    if metric not in AVAILABLE_METRICS:
        raise ValidationError(
            "target_metric",
            'Unknown target metric : "%s". Must be in %s' % (metric, AVAILABLE_METRICS),
        )
    else:
        return True


def validate_dataset(data):
    """
    Validate user given dataset
    """
    if data:
        if not isinstance(data, DatasetAutoFolds):
            raise ValidationError(
                "data",
                'Unknown data format. Must be and instance of "DatasetAutoFolds". Got "%s"'
                % type(data),
            )
        else:
            return True
    else:
        raise ValidationError("data", "Data must be present. Got: %s" % data)


def validate_max_evals(max_evals):
    """
    Max evaluations for hyperparameter tuning. Must be an unsigned integer greater than 0
    """
    if not isinstance(max_evals, int) or not max_evals > 0:
        raise ValidationError(
            "max_evals",
            "max_evals must be an unsigned integer greater than 0. Got {0}".format(
                max_evals
            ),
        )
    else:
        return True

def validate_cpu_time_limit(cpu_time_limit):
    """
    Execution time limit. Must be an unsigned integer greater than 0
    """
    if not isinstance(cpu_time_limit, int) or not cpu_time_limit > 0:
        raise ValidationError(
            "cpu_time_limit",
            "cpu_time_limit must be an unsigned integer greater than 0. Got {0}".format(
                cpu_time_limit
            ),
        )
    else:
        return True
