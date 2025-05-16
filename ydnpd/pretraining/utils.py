import os
import random
import hashlib
from typing import Optional

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve, classification_report

from ydnpd import load_dataset
from ydnpd.pretraining.consts import TEST_PROP, VAL_PROP, RANDOM_STATE
from ydnpd.harness.config import EVALUATION_KWARGS

SMALL_CONSTANT = 1e-12


def get_seed_from_config(config):
    items_str = str(sorted(config.items()))
    return int(hashlib.sha256(items_str.encode()).hexdigest()[:8], 16)


def set_reproducibility(seed=RANDOM_STATE, deterministic=True, cublas_workspace_config=":4096:8"):
    """
    Set up environment for reproducibility.

    Args:
        seed (int): Seed for random number generators
        deterministic (bool): Whether to force deterministic behavior
        cublas_workspace_config (str): CUBLAS workspace configuration
            ":4096:8" - More memory (24MB) but possibly better performance
            ":16:8" - Less memory but may limit performance
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Set CUBLAS workspace config before creating CUDA tensors
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = cublas_workspace_config

        # Ensure one stream per handle for determinism
        torch.cuda.set_device(torch.cuda.current_device())

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    return {
        'Python seed': seed,
        'NumPy seed': np.random.get_state()[1][0],
        'PyTorch seed': torch.initial_seed(),
        'CUDA seed': torch.cuda.initial_seed() if torch.cuda.is_available() else None,
        'Deterministic': deterministic,
        'CUDNN benchmark': torch.backends.cudnn.benchmark,
        'CUDNN deterministic': torch.backends.cudnn.deterministic,
        'CUBLAS workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'not set')
    }


def set_strict_reproducibility_by_config(config, deterministic=True):
    return set_reproducibility(seed=get_seed_from_config(config),
                               deterministic=deterministic)


def preprocess_data(df, schema, target_column='y',
                    test_prop=TEST_PROP,
                    random_state=RANDOM_STATE):

    feature_columns = [col for col in df.columns if col != target_column]
    y = df[target_column].values
    X = df[feature_columns]

    cat_features = [col for col in feature_columns if schema[col]['type'] == 'categorical']
    cont_features = [col for col in feature_columns if schema[col]['type'] == 'continuous']

    print("Features to process:", (set(cat_features) | set(cont_features)))
    print("Available columns:", X.columns)

    missing = (set(cat_features) | set(cont_features)) - set(X.columns)
    if missing:
        raise KeyError(f"Missing features after transformation: {missing}")

    X_cat = X[cat_features].copy()
    X_cont = X[cont_features].copy()

    # Print original values before encoding
    for col in cat_features:
        print(f"{col} original values:", sorted(X_cat[col].unique()))

    print(f"{schema=}")

    # Make categories 0 to k-1
    for column in cat_features:
        value_map = {v: i for i, v in enumerate(schema[column]['values'])}
        X_cat.loc[:, column] = X_cat[column].map(value_map)
        print(f"{column} encoded values:", sorted(X_cat[column].unique()))

    cat_cardinalities = [len(schema[col]['values']) for col in cat_features]

    print("Categorical cardinalities:", cat_cardinalities)

    if X_cont.shape[1] > 0:
        X_cont = (X_cont - X_cont.mean()) / (X_cont.std() + SMALL_CONSTANT)

    X_cat_train, X_cat_valid, X_cont_train, X_cont_test, y_train, y_test = train_test_split(
        X_cat, X_cont, y, test_size=test_prop, random_state=random_state
    )

    print(f"cat training features shape: {X_cat_train.shape}")
    print(f"cont training features shape: {X_cont_train.shape}")
    print(f"cat val features shape: {X_cat_valid.shape}")
    print(f"cont test features shape: {X_cont_test.shape}")
    print(f"training targets shape: {y_train.shape}")
    print(f"test targets shape: {y_test.shape}")

    return (
        torch.tensor(X_cat_train.to_numpy(), dtype=torch.long),
        torch.tensor(X_cont_train.to_numpy(), dtype=torch.float32),
        torch.tensor(X_cat_valid.to_numpy(), dtype=torch.long),
        torch.tensor(X_cont_test.to_numpy(), dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        cat_cardinalities,
        schema
    )


def load_data_for_classification(dataset_pointer: str | tuple,
                                 subsampling: Optional[float],
                                 random_state: int = RANDOM_STATE):

    if not (subsampling is None or 0 < subsampling < 1):
        raise ValueError(f"`subsampling` should be either None or a float between 0 and 1 (not including), and not `{subsampling}")

    if isinstance(dataset_pointer, (tuple, list)):
        dataset_name, dataset_path = dataset_pointer
    else:
        dataset_name, dataset_path = dataset_pointer, None

    dataset_family, _ = dataset_name.split("/")
    if dataset_family not in {"acs", "edad", "we"}:
        raise ValueError("Only ACS, EDAD, and WE datasets is supported for classification")

    dataset, schema, _ = load_dataset(dataset_name, path=dataset_path)

    dataset = dataset.copy()
    column_ordered = sorted(schema.keys())
    assert set(dataset.columns) == set(column_ordered)
    dataset = dataset[column_ordered]
    assert (dataset.columns == column_ordered).all()

    target_col_name = EVALUATION_KWARGS[dataset_family]["classification_target_column"]

    if dataset_family == "acs":
        # "0": "Group quarters",
        # "1": "Own housing unit",
        # "2": "Rent housing unit"
        # Definition. The Census Bureau classifies all people not living in housing units as living in group quarters. A group quarters is a place where people live or stay, in a group living arrangement, that is owned or managed by an entity or organization providing housing and/or services for the residents.
        dataset[dataset[target_col_name] == 0] = 2
        dataset.loc[:, 'y'] = dataset[target_col_name] - 1  # Map 1,2 to 0,1 directly

    elif dataset_family == "edad":
        dataset.loc[:, 'y'] = dataset[target_col_name] - 1  # Map 1,2 to 0,1 directly

    elif dataset_family == "we":
        dataset.loc[:, 'y'] = dataset[target_col_name]

    assert set(dataset["y"].unique()) == {0, 1}

    # Stratify
    df_pos = dataset[dataset['y'] == 1]
    df_neg = dataset[dataset['y'] == 0]
    min_size = min(len(df_pos), len(df_neg))
    df_pos = df_pos.sample(n=min_size, random_state=random_state)
    df_neg = df_neg.sample(n=min_size, random_state=random_state)

    dataset = pd.concat([df_pos, df_neg])
    dataset = dataset.drop(target_col_name, axis=1)

    frac = 1 if subsampling is None else subsampling
    dataset = dataset.sample(frac=frac, random_state=random_state).reset_index(drop=True)

    schema_without_target = {col: col_schema for col, col_schema in schema.items()
                             if col != target_col_name}

    cat_schema = {
        col: col_schema | {"type": "categorical"}
        for col, col_schema in schema_without_target.items()
        if 'values' in col_schema
    }

    assert set(cat_schema.keys()) | {"y"} == set(dataset.columns)
    dataset = preprocess_data(dataset, cat_schema)

    return dataset


def print_model_performance(classifier, X_cat_test, X_cont_test, y_test):
    y_pred_prob = classifier.predict_proba(X_cat_test, X_cont_test)[:, 1]

    # roc curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

    # compute an f-score for the optimal threshold
    fscore = (2 * precision * recall) / (precision + recall + 1e-8)  # Avoid division by zero
    ix = np.argmax(fscore)
    optimal_threshold = thresholds[ix]

    # pred using the optimal threshold
    y_pred = classifier.predict(X_cat_test, X_cont_test, binary=True, threshold=optimal_threshold)

    # calc auc
    auc = roc_auc_score(y_test, y_pred_prob)

    # print results
    print()
    print(f"AUC: {auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(classification_report(y_test, y_pred))
    print()


def split_train_val(X_cat_train_val, X_cont_train_val, y_train_val):

    assert X_cont_train_val.shape[1] == 0

    X_cat_train, X_cat_val, y_train, y_val = train_test_split(X_cat_train_val,
                                                              y_train_val,
                                                              test_size=VAL_PROP,
                                                              random_state=RANDOM_STATE)

    X_cont_train, X_cont_val, y_train, y_val = train_test_split(X_cont_train_val,
                                                                y_train_val,
                                                                test_size=VAL_PROP,
                                                                random_state=RANDOM_STATE)

    return X_cat_train, X_cat_val, X_cont_train, X_cont_val, y_train, y_val
