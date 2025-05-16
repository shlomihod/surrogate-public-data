import itertools as it

from ydnpd.harness.experiment import Experiments

import copy

EPSILONS = [1, 2, 4, 8, 16]

NUM_RUNS = 10

FIXED_PREPROCESSOR_EPSILON = 0.0

SYNTHESIZERS = ["aim_jax", "privbayes", "gem"]

HPARAMS_DIMS = {
    # this is probably too many parameter combinations...
    "gem": {
        "preprocessor_eps": [FIXED_PREPROCESSOR_EPSILON],
        "k": [2, 3],
        "T": [50, 100],
        "alpha": [0.1, 0.5],
        "ema_weights_beta": [0.1, 0.9]
    },
    "aim": {
        "preprocessor_eps": [FIXED_PREPROCESSOR_EPSILON],
        "degree": [2, 3],
        "rounds": [20, 40],
    },
    "mwem": {
        "preprocessor_eps": [FIXED_PREPROCESSOR_EPSILON],
        "q_count": [512, 1024],  # [128, 512],
        "marginal_width": [2, 3],
        "iterations": [5, 10, 50],
        "add_ranges": [False, True],
        "split_factor": [1, 2, 3, 4],  # , None],
        "mult_weights_iterations": [5, 10, 20],
    },
    "mst": {
        "preprocessor_eps": [FIXED_PREPROCESSOR_EPSILON],
    },
    "privbayes": {
        "theta": [2, 8, 32, 64],
        "epsilon_split": [0.1, 0.5, 0.75],
    },
    "patectgan": {
        "preprocessor_eps": [FIXED_PREPROCESSOR_EPSILON],
        "embedding_dim": [128],
        "generator_dim": [(256, 256)],
        "discriminator_dim": [(256, 256)],
        "epochs": [300],
        "generator_lr": [2e-4, 2e-5],
        "discriminator_lr": [2e-4, 2e-5],
        "generator_decay": [1e-6],
        "discriminator_decay": [1e-6],
        "batch_size": [500],
        "noise_multiplier": [1e-3, 0.1, 1, 5],
        "loss": ["cross_entropy", "wasserstein"],
        "teacher_iters": [5],
        "student_iters": [5],
        "sample_per_teacher": [1000],
        "delta": [None],
        "moments_order": [100],
        # discriminator_steps: NOT BEING USED
    },
}

HPARAMS_DIMS['aim_torch'] = copy.copy(HPARAMS_DIMS['aim'])
HPARAMS_DIMS['aim_jax'] = copy.copy(HPARAMS_DIMS['aim'])

ALL_EXPERIMENTS = {
    "acs": Experiments(
        "acs/national",
        [
            "acs/national",
            "acs/massachusetts_upsampled",
            "acs/baseline_univariate",
            "acs/baseline_domain",
            "acs/arbitrary",
        ],
    ),
    "edad": Experiments(
        "edad/2023",
        [
            "edad/2023",
            "edad/2020",
            "edad/baseline_univariate",
            "edad/baseline_domain",
            "edad/arbitrary",
        ],
    ),
    "we": Experiments(
        "we/2023",
        [
            "we/2023",
            "we/2018",
            "we/baseline_univariate",
            "we/baseline_domain",
            "we/arbitrary",
        ],
    ),
}

DATASET_NAMES = set(
    it.chain(
        *(
            [experiments.test_name] + experiments.dev_names
            for experiments in ALL_EXPERIMENTS.values()
        )
    )
)

EVALUATION_KWARGS = {
    "_": {
        "marginals_k": 3,
    },
    "acs": {
        "classification_target_column": "OWN_RENT",
    },
    "edad": {
        "classification_target_column": "MOVI_21_1",
    },
    "we": {
        "classification_target_column": "how_long_have_you_worked_for_your_current_employer",
    },
}
