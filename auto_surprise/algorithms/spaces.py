"""
Spaces defined for the algorithms to user
"""

from hyperopt import hp

SVD_DEFAULT_SPACE = {
    'n_factors': hp.choice('n_factors', range(1, 100)),
    'n_epochs': hp.choice('n_epochs', range(5, 200)),
    'lr_bu': hp.uniform('lr_bu', 0.0001, 0.5),
    'lr_bi': hp.uniform('lr_bi', 0.0001, 0.5),
    'lr_pu': hp.uniform('lr_pu', 0.0001, 0.5),
    'lr_qi': hp.uniform('lr_qi', 0.0001, 0.5),
    'reg_bu': hp.uniform('reg_bu', 0.0001, 0.5),
    'reg_bi': hp.uniform('reg_bi', 0.0001, 0.5),
    'reg_pu': hp.uniform('reg_pu', 0.0001, 0.5),
    'reg_qi': hp.uniform('reg_qi', 0.0001, 0.5)
}

SVDPP_SPACE = {
    **SVD_DEFAULT_SPACE,
    'lr_yj': hp.uniform('lr_yj', 0.0001, 0.5),
    'reg_yj': hp.uniform('reg_yj', 0.0001, 0.5)
}

SIMILARITY_OPTIONS_SPACE = {
    'name': hp.choice('name', ['cosine', 'msd', 'pearson', 'pearson_baseline']),
    'user_based': hp.choice('user_based', [False, True]),
    'min_support': hp.choice('min_support', range(1, 100)),
    'shrinkage': hp.choice('shrinkage', range(1, 300))
}

BSL_OPTIONS_SPACE = hp.choice('bsl_options', [
    {
        'method': 'als',
        'reg_i': hp.choice('reg_i', range(1, 100)),
        'reg_u': hp.choice('reg_u', range(1, 100)),
        'n_epochs': hp.choice('n_epochs', range(5, 200)),
    },
    {
        'method': 'sgd',
        'reg': hp.uniform('reg', 0.0001, 0.5),
        'learning_rate': hp.uniform('learning_rate', 0.0001, 0.5)
    }
])

KNN_DEFAULT_SPACE = {
    'k': hp.choice('k', range(1, 500)),
    'min_k': hp.choice('min_k', range(1, 10)),
    'sim_options': SIMILARITY_OPTIONS_SPACE
}

KNN_BASELINE_SPACE = {
    **KNN_DEFAULT_SPACE,
    'bsl_options': BSL_OPTIONS_SPACE
}

NMF_DEFAULT_SPACE = {
    'n_factors': hp.choice('n_factors', range(1, 100)),
    'n_epochs': hp.choice('n_epochs', range(5, 200)),
    'lr_bu': hp.uniform('lr_bu', 0.0001, 0.5),
    'lr_bi': hp.uniform('lr_bi', 0.0001, 0.5),
    'reg_bu': hp.uniform('reg_bu', 0.0001, 0.5),
    'reg_bi': hp.uniform('reg_bi', 0.0001, 0.5),
    'reg_pu': hp.uniform('reg_pu', 0.0001, 0.5),
    'reg_qi': hp.uniform('reg_qi', 0.0001, 0.5),
    'biased': hp.choice('biased', [False, True])
}

CO_CLUSTERING_DEFAULT_SPACE = {
    'n_cltr_u': hp.choice('n_cltr_u', range(1, 100)),
    'n_cltr_i': hp.choice('n_cltr_i', range(1, 100)),
    'n_epochs': hp.choice('n_epochs', range(5, 200)),
}
