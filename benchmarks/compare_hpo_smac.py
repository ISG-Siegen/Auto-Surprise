from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise import Dataset
from surprise import Reader
import numpy as np
import pandas as pd
import os

from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

SVD_DEFAULT_SPACE = {
    'n_factors': hp.choice('n_factors', range(1, 100)),
    'n_epochs': hp.choice('n_epochs', range(1, 100)),
    'lr_all': hp.loguniform('lr_all', np.log(0.0001), np.log(0.1)),
    'reg_all': hp.loguniform('reg_all', np.log(0.0001), np.log(0.1)),
}

SVD_SMAC_SPACE = {
    'n_factors': UniformIntegerHyperparameter("n_factors", 1, 100),
    'n_epochs': UniformIntegerHyperparameter("n_epochs", 1, 100),
    'lr_all': UniformFloatHyperparameter("lr_all", 0.0001, 0.1, log=True),
    'reg_all': UniformFloatHyperparameter("reg_all", 0.0001, 0.1, log=True),
}

file_path = os.path.expanduser('../datasets/ml-100k/u.data')
reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(1, 5))

data = Dataset.load_from_file(file_path, reader=reader)

bench_res = {
    'loss': []
}

def _hyperopt(params):
    algo = SVD(**params)
    return cross_validate(algo, data, measures=['RMSE', 'MAE', 'MSE'], cv=3, n_jobs=1)['test_rmse'].mean()

def _objective(params):
    loss = _hyperopt(params)

    bench_res['loss'].append(loss)

    return {
        'loss': loss,
        'status': STATUS_OK,
        'hyperparams': params
    }

def best_hyperparams():
    trials = Trials()
    best = fmin(
        _objective,
        SVD_DEFAULT_SPACE,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )
    
    return best

def best_hyperparams_smac():
    iteration = 1
    cs = ConfigurationSpace()
    cs.add_hyperparameters(SVD_SMAC_SPACE.values())
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                     "runcount-limit": 100,  # max. number of function evaluations; for this example set to a low number
                     "cs": cs,  # configuration space
                     "deterministic": "true"
                     })
    smac = SMAC4HPO(scenario=scenario,
               rng=np.random.RandomState(42),
               tae_runner=_hyperopt,
               )
    smac.optimize()


best_hyperparams()

results = pd.DataFrame.from_dict(bench_res)
print(results)
results.to_csv('hpo.csv')

# best_hyperparams_smac()
