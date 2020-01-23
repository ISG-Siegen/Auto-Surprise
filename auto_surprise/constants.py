ALGORITHM_MAP = {
    'svd': 'AutoSurpriseSVD',
    'svdpp': 'AutoSurpriseSVDpp',
    'nmf': 'AutoSurpriseNMF',
    'knn_baseline': 'AutoSurpriseKNNBaseline',
    'knn_basic': 'AutoSurpriseKNNBasic',
    'knn_with_means': 'AutoSurpriseKNNWithMeans',
    'knn_with_z_score': 'AutoSurpriseKNNWithZScore',
    'co_clustering': 'AutoSurpriseCoClustering',
    'slope_one': 'AutoSurpriseSlopeOne',
    'baseline_only': 'AutoSurpriseBaselineOnly',
    'normal_predictor': 'AutoSurpriseNormalPredictor',
}

FULL_ALGO_LIST = ['svd', 'svdpp', 'nmf', 'knn_baseline', 'knn_with_means', 'knn_with_z_score', 'co_clustering', 'slope_one', 'baseline_only', 'normal_predictor']
QUICK_COMPUTE_ALGO_LIST = ['svd', 'nmf', 'knn_baseline', 'knn_with_means', 'knn_with_z_score', 'co_clustering', 'slope_one', 'baseline_only', 'normal_predictor']

DEFAULT_TARGET_METRIC = 'test_rmse'

AVAILABLE_METRICS = [
    'test_rmse',
    'test_mae',
    'test_mse',
]

ACCURACY_METRICS = [
    'RMSE',
    'MAE',
    'MSE',
]

DEFAULT_MAX_EVALS = 15

MAX_WORKERS = 20

CV_N_JOBS = -1
