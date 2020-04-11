"""
Spaces defined for the algorithms to user
"""

from hyperopt import hp

SVD_DEFAULT_SPACE = {
    'n_factors': hp.choice('n_factors', range(1, 100)),
    'n_epochs': hp.choice('n_epochs', range(5, 100)),
    'lr_bu': hp.uniform('lr_bu', 0.0001, 0.3),
    'lr_bi': hp.uniform('lr_bi', 0.0001, 0.3),
    'lr_pu': hp.uniform('lr_pu', 0.0001, 0.3),
    'lr_qi': hp.uniform('lr_qi', 0.0001, 0.3),
    'reg_bu': hp.uniform('reg_bu', 0.0001, 0.3),
    'reg_bi': hp.uniform('reg_bi', 0.0001, 0.3),
    'reg_pu': hp.uniform('reg_pu', 0.0001, 0.3),
    'reg_qi': hp.uniform('reg_qi', 0.0001, 0.3)
}

SVDPP_SPACE = {
    **SVD_DEFAULT_SPACE,
    'lr_yj': hp.uniform('lr_yj', 0.0001, 0.3),
    'reg_yj': hp.uniform('reg_yj', 0.0001, 0.3)
}

SIMILARITY_OPTIONS_SPACE = {
    'name': hp.choice('name', ['cosine', 'msd', 'pearson', 'pearson_baseline']),
    'user_based': hp.choice('user_based', [False, True]),
    'shrinkage': hp.choice('shrinkage', range(1, 300))
}

KNN_DEFAULT_SPACE = {
    'k': hp.choice('k', range(1, 100)),
    'min_k': hp.choice('min_k', range(1, 10)),
    'sim_options': SIMILARITY_OPTIONS_SPACE
}

NMF_DEFAULT_SPACE = {
    'n_factors': hp.choice('n_factors', range(1, 100)),
    'n_epochs': hp.choice('n_epochs', range(5, 100)),
    'lr_bu': hp.uniform('lr_bu', 0.0001, 0.3),
    'lr_bi': hp.uniform('lr_bi', 0.0001, 0.3),
    'reg_bu': hp.uniform('reg_bu', 0.0001, 0.3),
    'reg_bi': hp.uniform('reg_bi', 0.0001, 0.3),
    'reg_pu': hp.uniform('reg_pu', 0.0001, 0.3),
    'reg_qi': hp.uniform('reg_qi', 0.0001, 0.3),
    'biased': hp.choice('biased', [False, True])
}

CO_CLUSTERING_DEFAULT_SPACE = {
    'n_cltr_u': hp.choice('n_cltr_u', range(1, 30)),
    'n_cltr_i': hp.choice('n_cltr_i', range(1, 30)),
    'n_epochs': hp.choice('n_epochs', range(5, 100)),
}
