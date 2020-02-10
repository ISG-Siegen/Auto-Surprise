from hyperopt import hp

SVD_DEFAULT_SPACE = {
    'n_factors': hp.choice('n_factors', range(1, 100)),
    'n_epochs': hp.choice('n_epochs', range(1, 20)),
    'lr_all': hp.uniform('lr_all', 0.001, 0.3),
    'reg_all': hp.uniform('reg_all', 0.001, 0.3),
}

SIMILARITY_OPTIONS_SPACE = {
    'name': hp.choice('name', ['cosine', 'msd', 'pearson', 'pearson_baseline']),
    'user_based': hp.choice('user_based', [False, True]),
    'shrinkage': hp.choice('shrinkage', range(1, 300))
}

KNN_DEFAULT_SPACE = {
    'k': hp.choice('k', range(1,100)),
    'min_k': hp.choice('min_k', range(1,10)),
}

NMF_DEFAULT_SPACE = {
    'n_factors': hp.choice('n_factors', range(1, 100)),
    'n_epochs': hp.choice('n_epochs', range(1, 20)),
}

CO_CLUSTERING_DEFAULT_SPACE = {
    'n_cltr_u': hp.choice('n_cltr_u', range(1,30)),
    'n_cltr_i': hp.choice('n_cltr_i', range(1,30)),
    'n_epochs': hp.choice('n_epochs', range(5,50)),
}
