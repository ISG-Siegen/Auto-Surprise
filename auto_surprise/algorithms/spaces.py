"""
Spaces defined for the algorithms to user
"""
from math import log
from hyperopt import hp

LR_LOG_LOWER = log(0.0001)
LR_LOG_UPPER = log(0.1)

SVD_DEFAULT_SPACE = {
    'n_factors': hp.choice('n_factors', range(1, 500)),
    'n_epochs': hp.choice('n_epochs', range(1, 200)),
    'lr_bu': hp.loguniform('lr_bu', LR_LOG_LOWER, LR_LOG_UPPER),
    'lr_bi': hp.loguniform('lr_bi', LR_LOG_LOWER, LR_LOG_UPPER),
    'lr_pu': hp.loguniform('lr_pu', LR_LOG_LOWER, LR_LOG_UPPER),
    'lr_qi': hp.loguniform('lr_qi', LR_LOG_LOWER, LR_LOG_UPPER),
    'reg_bu': hp.loguniform('reg_bu', LR_LOG_LOWER, LR_LOG_UPPER),
    'reg_bi': hp.loguniform('reg_bi', LR_LOG_LOWER, LR_LOG_UPPER),
    'reg_pu': hp.loguniform('reg_pu', LR_LOG_LOWER, LR_LOG_UPPER),
    'reg_qi': hp.loguniform('reg_qi', LR_LOG_LOWER, LR_LOG_UPPER)
}

SVDPP_SPACE = {
    **SVD_DEFAULT_SPACE,
    'lr_yj': hp.loguniform('lr_yj', LR_LOG_LOWER, LR_LOG_UPPER),
    'reg_yj': hp.loguniform('reg_yj', LR_LOG_LOWER, LR_LOG_UPPER)
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
        'reg': hp.loguniform('reg', LR_LOG_LOWER, LR_LOG_UPPER),
        'learning_rate': hp.loguniform('learning_rate', LR_LOG_LOWER, LR_LOG_UPPER)
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
    'lr_bu': hp.loguniform('lr_bu', LR_LOG_LOWER, LR_LOG_UPPER),
    'lr_bi': hp.loguniform('lr_bi', LR_LOG_LOWER, LR_LOG_UPPER),
    'reg_bu': hp.loguniform('reg_bu', LR_LOG_LOWER, LR_LOG_UPPER),
    'reg_bi': hp.loguniform('reg_bi', LR_LOG_LOWER, LR_LOG_UPPER),
    'reg_pu': hp.loguniform('reg_pu', LR_LOG_LOWER, LR_LOG_UPPER),
    'reg_qi': hp.loguniform('reg_qi', LR_LOG_LOWER, LR_LOG_UPPER),
    'biased': hp.choice('biased', [False, True])
}

CO_CLUSTERING_DEFAULT_SPACE = {
    'n_cltr_u': hp.choice('n_cltr_u', range(1, 1000)),
    'n_cltr_i': hp.choice('n_cltr_i', range(1, 100)),
    'n_epochs': hp.choice('n_epochs', range(5, 200)),
}
