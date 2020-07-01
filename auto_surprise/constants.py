from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import KNNBaseline
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import SlopeOne
from surprise import CoClustering
from surprise import NormalPredictor
from surprise import BaselineOnly
from hyperopt import tpe

### Algorithm maps

ALGORITHM_MAP = {
    "svd": "AutoSurpriseSVD",
    "svdpp": "AutoSurpriseSVDpp",
    "nmf": "AutoSurpriseNMF",
    "knn_baseline": "AutoSurpriseKNNBaseline",
    "knn_basic": "AutoSurpriseKNNBasic",
    "knn_with_means": "AutoSurpriseKNNWithMeans",
    "knn_with_z_score": "AutoSurpriseKNNWithZScore",
    "co_clustering": "AutoSurpriseCoClustering",
    "slope_one": "AutoSurpriseSlopeOne",
    "baseline_only": "AutoSurpriseBaselineOnly",
    "normal_predictor": "AutoSurpriseNormalPredictor",
}

SURPRISE_ALGORITHM_MAP = {
    "svd": SVD,
    "svdpp": SVDpp,
    "nmf": NMF,
    "knn_baseline": KNNBaseline,
    "knn_basic": KNNBasic,
    "knn_with_means": KNNWithMeans,
    "knn_with_z_score": KNNWithZScore,
    "co_clustering": CoClustering,
    "slope_one": SlopeOne,
    "baseline_only": BaselineOnly,
    "normal_predictor": NormalPredictor,
}

FULL_ALGO_LIST = [
    "svd",
    "svdpp",
    "nmf",
    "knn_basic",
    "knn_baseline",
    "knn_with_means",
    "knn_with_z_score",
    "co_clustering",
    "slope_one",
    "baseline_only",
]
QUICK_COMPUTE_ALGO_LIST = [
    "svd",
    "nmf",
    "knn_basic",
    "knn_baseline",
    "knn_with_means",
    "knn_with_z_score",
    "co_clustering",
    "slope_one",
    "baseline_only",
]
BASELINE_ALGO = "normal_predictor"

### Metrics

DEFAULT_TARGET_METRIC = "test_rmse"

AVAILABLE_METRICS = [
    "test_rmse",
    "test_mae",
    "test_mse",
]

ACCURACY_METRICS = [
    "RMSE",
    "MAE",
    "MSE",
]

### Run params
DEFAULT_CV_ITERS = 5
DEFAULT_MAX_EVALS = 10
EVALS_MULTIPLIER = 5
MAX_WORKERS = None
DEFAULT_HPO_ALGO = tpe.suggest
CV_N_JOBS = 1
