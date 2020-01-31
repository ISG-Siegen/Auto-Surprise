from surprise.model_selection import cross_validate

# import algorithms
from auto_surprise.algorithms.auto_surprise_svd import AutoSurpriseSVD
from auto_surprise.algorithms.auto_surprise_svdpp import AutoSurpriseSVDpp
from auto_surprise.algorithms.auto_surprise_nmf import AutoSurpriseNMF
from auto_surprise.algorithms.auto_surprise_knn_baseline import AutoSurpriseKNNBaseline
from auto_surprise.algorithms.auto_surprise_knn_basic import AutoSurpriseKNNBasic
from auto_surprise.algorithms.auto_surprise_knn_with_means import AutoSurpriseKNNWithMeans
from auto_surprise.algorithms.auto_surprise_knn_with_z_score import AutoSurpriseKNNWithZScore
from auto_surprise.algorithms.auto_surprise_slope_one import AutoSurpriseSlopeOne
from auto_surprise.algorithms.auto_surprise_co_clustering import AutoSurpriseCoClustering
from auto_surprise.algorithms.auto_surprise_normal_predictor import AutoSurpriseNormalPredictor
from auto_surprise.algorithms.auto_surprise_baseline_only import AutoSurpriseBaselineOnly

from auto_surprise.constants import ALGORITHM_MAP, DEFAULT_TARGET_METRIC, CV_N_JOBS
import sys

class Trainer(object):
    def __init__(self, algo=None, data=None, target_metric=DEFAULT_TARGET_METRIC, debug=False):
        """
        Initialize new trainer
        """
        self._debug = debug
        # Dynamically instantiate algorithm
        self.algo = getattr(sys.modules[__name__], ALGORITHM_MAP[algo])(data=data, metric=target_metric, cv_n_jobs=CV_N_JOBS, debug=debug)

    def start(self, max_evals):
        try:
            best, trials = self.algo.best_hyperparams(max_evals=max_evals)

            best_trial = None
            # Sort best trial based on loss value
            if best:
                best_trial = sorted(trials.results, key=lambda x: x['loss'], reverse=False)[0]
            else:
                best_trial = trials

            if self._debug:
                print("Best: ", best)

            return best, best_trial
        except Exception as e:
            print('Exception : ', e)

            if self._debug
                raise

            return False, False
