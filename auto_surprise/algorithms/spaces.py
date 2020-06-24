"""
Spaces defined for the algorithms to user
It might be a good idea to experiment with some of the distributions here
"""

from math import log
from hyperopt import hp
from hyperopt.pyll.base import scope

LR_LOG_LOWER = log(0.0001)
LR_LOG_UPPER = log(0.1)

# NOTE: Do not use keyword arguments for hyperopt distributions. It causes a crash when using ATPE

SVD_DEFAULT_SPACE = {
    "n_factors": scope.int(hp.quniform("n_factors", 1, 200, 1)),
    "n_epochs": scope.int(hp.quniform("n_epochs", 1, 200, 1)),
    "lr_bu": hp.loguniform("lr_bu", LR_LOG_LOWER, LR_LOG_UPPER),
    "lr_bi": hp.loguniform("lr_bi", LR_LOG_LOWER, LR_LOG_UPPER),
    "lr_pu": hp.loguniform("lr_pu", LR_LOG_LOWER, LR_LOG_UPPER),
    "lr_qi": hp.loguniform("lr_qi", LR_LOG_LOWER, LR_LOG_UPPER),
    "reg_bu": hp.loguniform("reg_bu", LR_LOG_LOWER, LR_LOG_UPPER),
    "reg_bi": hp.loguniform("reg_bi", LR_LOG_LOWER, LR_LOG_UPPER),
    "reg_pu": hp.loguniform("reg_pu", LR_LOG_LOWER, LR_LOG_UPPER),
    "reg_qi": hp.loguniform("reg_qi", LR_LOG_LOWER, LR_LOG_UPPER),
}

SVDPP_SPACE = {
    **SVD_DEFAULT_SPACE,
    "lr_yj": hp.loguniform("lr_yj", LR_LOG_LOWER, LR_LOG_UPPER),
    "reg_yj": hp.loguniform("reg_yj", LR_LOG_LOWER, LR_LOG_UPPER),
}

# TODO: Shrinkage param is only used when pearson_baseline is selected. It would
# be great if we could conditionally optimize if pearson_baseline is used
SIMILARITY_OPTIONS_SPACE = {
    "name": hp.choice("name", ["cosine", "msd", "pearson", "pearson_baseline"]),
    "user_based": hp.choice("user_based", [False, True]),
    "min_support": scope.int(hp.quniform("min_support", 1, 100, 1)),
    "shrinkage": scope.int(hp.quniform("shrinkage", 1, 300, 1)),
}

BSL_OPTIONS_SPACE = hp.choice(
    "bsl_options",
    [
        {
            "method": "als",
            "reg_i": hp.uniform("reg_i", 1, 100),
            "reg_u": hp.uniform("reg_u", 1, 100),
            "n_epochs": scope.int(hp.quniform("n_epochs", 5, 200, 1)),
        },
        {
            "method": "sgd",
            "reg": hp.loguniform("reg", LR_LOG_LOWER, LR_LOG_UPPER),
            "learning_rate": hp.loguniform("learning_rate", LR_LOG_LOWER, LR_LOG_UPPER),
        },
    ],
)

BASELINE_ONLY_SPACE = {"bsl_options": BSL_OPTIONS_SPACE}

KNN_DEFAULT_SPACE = {
    "k": scope.int(hp.quniform("k", 1, 500, 1)),
    "min_k": scope.int(hp.quniform("min_k", 1, 50, 1)),
    "sim_options": SIMILARITY_OPTIONS_SPACE,
}

KNN_BASELINE_SPACE = {**KNN_DEFAULT_SPACE, "bsl_options": BSL_OPTIONS_SPACE}

NMF_DEFAULT_SPACE = {
    "n_factors": scope.int(hp.quniform("n_factors", 1, 500, 1)),
    "n_epochs": scope.int(hp.quniform("n_epochs", 5, 200, 1)),
    "lr_bu": hp.loguniform("lr_bu", LR_LOG_LOWER, LR_LOG_UPPER),
    "lr_bi": hp.loguniform("lr_bi", LR_LOG_LOWER, LR_LOG_UPPER),
    "reg_bu": hp.loguniform("reg_bu", LR_LOG_LOWER, LR_LOG_UPPER),
    "reg_bi": hp.loguniform("reg_bi", LR_LOG_LOWER, LR_LOG_UPPER),
    "reg_pu": hp.loguniform("reg_pu", LR_LOG_LOWER, LR_LOG_UPPER),
    "reg_qi": hp.loguniform("reg_qi", LR_LOG_LOWER, LR_LOG_UPPER),
    "biased": hp.choice("biased", [False, True]),
}

CO_CLUSTERING_DEFAULT_SPACE = {
    "n_cltr_u": scope.int(hp.quniform("n_cltr_u", 1, 1000, 1)),
    "n_cltr_i": scope.int(hp.quniform("n_cltr_i", 1, 100, 1)),
    "n_epochs": scope.int(hp.quniform("n_epochs", 5, 200, 1)),
}
