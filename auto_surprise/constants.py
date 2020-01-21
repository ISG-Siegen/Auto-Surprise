ALGORITHM_MAP = {
    'svd': 'AutoSurpriseSVD',
    'svdpp': 'AutoSurpriseSVDpp',
    'nmf': 'AutoSurpriseNMF',
    'knn_baseline': 'AutoSurpriseKNNBaseline',
    'knn_basic': 'AutoSurpriseKNNBasic',
    'knn_with_means': 'AutoSurpriseKNNWithMeans',
    'knn_with_z_score': 'AutoSurpriseKNNWithZScore',
    'co_clustering': 'AutoSurpriseCoClustering',
    'sope_one': 'AutoSurpriseSlopeOne',
    'baseline_only': 'AutoSurpriseBaselineOnly',
    'normal_predictor': 'AutoSurpriseNormalPredictor',
}

DEFAULT_TARGET_METRIC = 'test_rmse'

DEFAULT_MAX_EVALS = 15

MAX_WORKERS = 20
