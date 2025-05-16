import itertools as it
from math import comb

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats.contingency import association


EVALUATION_METRICS = [
    "total_variation_distance",
    "marginals_3_max_abs_diff_error",
    "marginals_3_avg_abs_diff_error",
    "thresholded_marginals_3_max_abs_diff_error",
    "thresholded_marginals_3_avg_abs_diff_error",
    "pearson_corr_max_abs_diff",
    "pearson_corr_avg_abs_diff",
    "cramer_v_corr_max_abs_diff",
    "cramer_v_corr_avg_abs_diff",
    "error_rate_diff",
    "error_rate_train_dataset",
    "error_rate_synth_dataset",
    "auc_diff",
    "auc_train_dataset",
    "auc_synth_dataset",
]


RANDOM_SEED = 42


def _safe_association(observed, method):

    # Check if either variable has only one value
    if observed.shape[0] == 1 or observed.shape[1] == 1:
        return 0.0

    # If we have multiple values, calculate association
    return association(observed, method=method)


def _asocciation_matrix(df, method):
    cols = df.columns
    n = len(cols)
    cramers_v_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                cramers_v_mat[i, j] = 1.0
            else:
                confusion_matrix = pd.crosstab(df[cols[i]], df[cols[j]])
                cramers_v_mat[i, j] = _safe_association(confusion_matrix, method)
    return pd.DataFrame(cramers_v_mat, index=cols, columns=cols)


def calc_distribution_distances(train_dataset: pd.DataFrame, synth_dataset: pd.DataFrame):
    train_dist = train_dataset.value_counts(normalize=True)
    synth_dist = synth_dataset.value_counts(normalize=True)

    train_dist, synth_dist = train_dist.align(synth_dist, fill_value=0)

    return {"total_variation_distance": ((train_dist - synth_dist)
                                         .abs()
                                         .sum()
                                         / 2)}

def calc_k_marginals_abs_diff_errors(
    train_dataset: pd.DataFrame, synth_dataset: pd.DataFrame, marginals_k: int
) -> dict[str, int]:

    columns = list(train_dataset.columns)

    marginals_abs_diff_errors = []

    def count_fn(dataset: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        return dataset.groupby(keys).size().to_frame("count").reset_index()

    for keys in it.combinations(columns, marginals_k):
        keys = list(keys)

        marginals_abs_diff_errors.extend(
            pd.merge(
                count_fn(train_dataset, keys),
                count_fn(synth_dataset, keys),
                how="outer",
                on=keys,
            )
            .fillna(0)
            .apply(lambda row: np.abs(row["count_x"] - row["count_y"]), axis=1)
            .to_list()
        )

    query_count = comb(len(columns), marginals_k)

    return {
        f"marginals_{marginals_k}_max_abs_diff_error": np.max(marginals_abs_diff_errors)
        / len(train_dataset),
        f"marginals_{marginals_k}_avg_abs_diff_error": np.sum(marginals_abs_diff_errors)
        / (len(train_dataset) * query_count),
    }


def calc_thresholded_marginals_k_abs_diff_errors(
    train_dataset: pd.DataFrame,
    synth_dataset: pd.DataFrame,
    schema: dict,
    marginals_k: int,
) -> dict[str, int]:
    datasets = [train_dataset.copy(), synth_dataset.copy()]

    for column in train_dataset.columns:
        column_schema = schema[column]
        if (values := list(column_schema.get("values"))) is not None:
            assert column_schema["dtype"].startswith(
                "int"
            ), "Only integer columns can be categorical; you might have missing values"

            assert all(
                isinstance(v, int) for v in values
            ), "Categorical values must be integers"
            assert len(values) == len(set(values)), "Categorical values must be unique"
            assert values == sorted(values), "Categorical values must be sorted"

            mid_value = values[len(values) // 2]

            for dataset in datasets:

                # if the value is less than the mid value, set it to 0, otherwise set it to 1
                dataset[column] = (dataset[column] < mid_value).astype(int)

        args = datasets + [marginals_k]

    return {
        f"thresholded_{k}": v
        for k, v in calc_k_marginals_abs_diff_errors(*args).items()
    }


def calc_classification_error_rate(
    train_dataset: pd.DataFrame,
    eval_dataset: pd.DataFrame,
    synth_dataset: pd.DataFrame,
    target_column: str,
) -> dict[str, float]:

    X_train, y_train = (
        train_dataset.drop(columns=[target_column]),
        train_dataset[target_column],
    )
    X_eval, y_eval = (
        eval_dataset.drop(columns=[target_column]),
        eval_dataset[target_column],
    )
    X_synth, y_synth = (
        synth_dataset.drop(columns=[target_column]),
        synth_dataset[target_column],
    )

    y_eval_np = y_eval.to_numpy()

    def _calc_stats(X, y):
        if y.nunique() > 1:
            model = GradientBoostingClassifier().fit(X, y)
            error_rate = 1 - model.score(X_eval, y_eval)
            y_pred_proba = model.predict_proba(X_eval)

            if len(np.unique(y_eval_np)) > 2:
                auc = 1 - roc_auc_score(
                    y_eval_np, y_pred_proba, multi_class="ovo"
                )
            else:
                auc = 1 - roc_auc_score(y_eval_np, y_pred_proba[:, 1])

        else:
            error_rate = np.nan
            auc = np.nan

        return error_rate, auc

    error_rate_train_dataset, auc_train_dataset= _calc_stats(X_train, y_train)
    error_rate_synth_dataset, auc_synth_dataset= _calc_stats(X_synth, y_synth)

    return {
        "error_rate_train_dataset": error_rate_train_dataset,
        "error_rate_synth_dataset": error_rate_synth_dataset,
        "error_rate_diff": error_rate_train_dataset - error_rate_synth_dataset,
        "auc_train_dataset": auc_train_dataset,
        "auc_synth_dataset": auc_synth_dataset,
        "auc_diff": auc_train_dataset - auc_synth_dataset,
    }


def calculate_corr(train_dataset: pd.DataFrame, synth_dataset: pd.DataFrame) -> dict:
    train_dataset_pearson_corr = np.tril(_asocciation_matrix(train_dataset, method="pearson"), k=-1)
    synth_dataset_pearson_corr = np.tril(_asocciation_matrix(synth_dataset, method="pearson"), k=-1)
    abs_diff_pearson_corr = np.abs(
        train_dataset_pearson_corr - synth_dataset_pearson_corr
    )

    train_dataset_cramer_v_corr = np.tril(_asocciation_matrix(train_dataset, method="cramer"), k=-1)
    synth_dataset_cramer_v_corr = np.tril(_asocciation_matrix(synth_dataset, method="cramer"), k=-1)
    abs_diff_cramer_v_corr = np.abs(
        train_dataset_cramer_v_corr - synth_dataset_cramer_v_corr
    )

    num_corrs = comb(len(train_dataset.columns), 2)

    return {
        "pearson_corr_train_dataset": train_dataset_pearson_corr,
        "pearson_corr_synth_dataset": synth_dataset_pearson_corr,
        "cramer_v_corr_train_dataset": train_dataset_cramer_v_corr,
        "cramer_v_corr_synth_dataset": synth_dataset_cramer_v_corr,
        "pearson_corr_max_abs_diff": np.max(abs_diff_pearson_corr),
        "pearson_corr_avg_abs_diff": np.sum(abs_diff_pearson_corr) / num_corrs,
        "cramer_v_corr_max_abs_diff": np.max(abs_diff_cramer_v_corr),
        "cramer_v_corr_avg_abs_diff": np.sum(abs_diff_cramer_v_corr) / num_corrs,
    }


def evaluate_two(
    train_dataset: pd.DataFrame,
    eval_dataset: pd.DataFrame,
    synth_dataset: pd.DataFrame,
    schema: dict,
    classification_target_column: str,
    marginals_k: int,
) -> dict:

    assert train_dataset.shape == synth_dataset.shape
    assert list(train_dataset.columns) == list(synth_dataset.columns)

    return (
        calc_distribution_distances(train_dataset, synth_dataset)
        | calc_k_marginals_abs_diff_errors(train_dataset, synth_dataset, marginals_k)
        | calc_thresholded_marginals_k_abs_diff_errors(
            train_dataset, synth_dataset, schema, marginals_k
        )
        | calculate_corr(train_dataset, synth_dataset)
        | calc_classification_error_rate(
            train_dataset,
            eval_dataset,
            synth_dataset,
            classification_target_column,
        )
    )
