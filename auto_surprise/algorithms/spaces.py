"""
Spaces defined for the algorithms to user
"""

from hyperopt import hp

SVD_DEFAULT_SPACE = {
    'n_factors': hp.choice('n_factors', range(1, 500)),
    'n_epochs': hp.choice('n_epochs', range(1, 200)),
    'lr_bu': hp.loguniform('lr_bu', 0.0001, 0.1),
    'lr_bi': hp.loguniform('lr_bi', 0.0001, 0.1),
    'lr_pu': hp.loguniform('lr_pu', 0.0001, 0.1),
    'lr_qi': hp.loguniform('lr_qi', 0.0001, 0.1),
    'reg_bu': hp.loguniform('reg_bu', 0.0001, 0.1),
    'reg_bi': hp.loguniform('reg_bi', 0.0001, 0.1),
    'reg_pu': hp.loguniform('reg_pu', 0.0001, 0.1),
    'reg_qi': hp.loguniform('reg_qi', 0.0001, 0.1)
}

SVDPP_SPACE = {
    **SVD_DEFAULT_SPACE,
    'lr_yj': hp.loguniform('lr_yj', 0.0001, 0.1),
    'reg_yj': hp.loguniform('reg_yj', 0.0001, 0.1)
}

SIMILARITY_OPTIONS_SPACE = {
    'name': hp.choice('name', ['cosine', 'msd', 'pearson', 'pearson_baseline']),
    'user_based': hp.choice('user_based', [False, True]),
    'min_support': hp.choice('min_support', range(1, 100)),
    # 'shrinkage': hp.choice('shrinkage', range(1, 300))
}

BSL_OPTIONS_SPACE = hp.choice('bsl_options', [
    {
        'method': 'als',
        'reg_i': hp.uniform('reg_i', 1, 100),
        'reg_u': hp.uniform('reg_u', 1, 100),
        'n_epochs': hp.choice('n_epochs', range(5, 200)),
    },
    {
        'method': 'sgd',
        'reg': hp.loguniform('reg', 0.0001, 0.1),
        'learning_rate': hp.loguniform('learning_rate', 0.0001, 0.1)
    }
])

KNN_DEFAULT_SPACE = {
    'k': hp.choice('k', range(1, 500)),
    'min_k': hp.choice('min_k', range(1, 30)),
    'sim_options': SIMILARITY_OPTIONS_SPACE
}

KNN_BASELINE_SPACE = {
    **KNN_DEFAULT_SPACE,
    'bsl_options': BSL_OPTIONS_SPACE
}

NMF_DEFAULT_SPACE = {
    'n_factors': hp.choice('n_factors', range(1, 500)),
    'n_epochs': hp.choice('n_epochs', range(5, 200)),
    'lr_bu': hp.loguniform('lr_bu', 0.0001, 0.1),
    'lr_bi': hp.loguniform('lr_bi', 0.0001, 0.1),
    'reg_bu': hp.loguniform('reg_bu', 0.0001, 0.1),
    'reg_bi': hp.loguniform('reg_bi', 0.0001, 0.1),
    'reg_pu': hp.loguniform('reg_pu', 0.0001, 0.1),
    'reg_qi': hp.loguniform('reg_qi', 0.0001, 0.1),
    'biased': hp.choice('biased', [False, True])
}

CO_CLUSTERING_DEFAULT_SPACE = {
    'n_cltr_u': hp.choice('n_cltr_u', range(1, 1000)),
    'n_cltr_i': hp.choice('n_cltr_i', range(1, 100)),
    'n_epochs': hp.choice('n_epochs', range(5, 200)),
}
