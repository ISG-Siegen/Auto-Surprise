from surprise.model_selection import cross_validate
from auto_surprise.algorithms.auto_surprise_svd import AutoSurpriseSVD
from auto_surprise.algorithms.auto_surprise_knn_basic import AutoSurpriseKNNBasic
from auto_surprise.constants import ALGORITHM_MAP
import sys

class Trainer(object):
    def __init__(self, algo=None, data=None, debug=False):
        """
        Initialize new trainer
        """
        self._debug = debug
        # Dynamically instantiate algorithm
        self.algo = getattr(sys.modules[__name__], ALGORITHM_MAP[algo])(data=data, debug=debug)

    def start(self):
        best, trials = self.algo.best_hyperparams()

        # Sort best trial based on loss value
        best_trial = sorted(trials.results, key=lambda x: x['loss'], reverse=False)[0]

        if self._debug:
            print("Best: ", best)

        return best, best_trial
