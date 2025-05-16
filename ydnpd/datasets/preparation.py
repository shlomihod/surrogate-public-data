import json
from pathlib import Path

import pandas as pd
import numpy as np

from ydnpd.datasets.loader import load_dataset, split_train_eval_datasets, DATA_ROOT
from ydnpd.datasets.arbitrary import RandomBayesianNetwork


RANDOM_SEED_GENERATION = 42
ARBITRARY_MAX_DEGREE = 5
ARBITRARY_ALPHA = 1


def save_dataset(dataset, data_path, name):
    data_path = Path(data_path)
    dataset_path = data_path / f"{name}.csv"
    print(dataset_path)
    dataset.to_csv(dataset_path, index=False)


def generate_baseline_domain(dataset, schema, null_prop=None):

    rng = np.random.default_rng(RANDOM_SEED_GENERATION)

    num_records = len(dataset)

    generated_dataset = {}

    for column in dataset.columns:

        if column not in schema:
            raise ValueError(f"Column '{column}' not found in schema")

        if "values" not in schema[column]:
            raise ValueError(f"Column '{column}' must have 'values' key")

        values = [v for v in schema[column]["values"]]

        if schema[column].get("has_null", False):
            null_value = schema[column]["null_value"]
            values.remove(null_value)

        sampled_values = rng.choice(values, num_records)

        if schema[column].get("has_null", False) and null_prop is not None:
            mask = rng.random(num_records) < null_prop
            sampled_values[mask] = schema[column]["null_value"]

        generated_dataset[column] = sampled_values

    return pd.DataFrame(generated_dataset)


def generate_baseline_univariate(dataset, schema, rounding):

    train_dataet, _ = split_train_eval_datasets(dataset)

    if rounding is None:
        df = pd.DataFrame(
            {
                column: (
                    values.sample(
                        n=len(dataset), replace=True, random_state=RANDOM_SEED_GENERATION
                    ).reset_index(drop=True)
                )
                for column, values in train_dataet.items()
            }
        )

    else:

        rng = np.random.default_rng(RANDOM_SEED_GENERATION)

        pseudo_probs = {column:
                        dataset[column]
                        .value_counts(normalize=True)
                        .round(rounding)
                        for column in dataset.columns}
        probs = {column: (pp / pp.sum()).to_dict() for column, pp in pseudo_probs.items()}
        df = pd.DataFrame(
            {
                column: (
                    rng.choice(
                        list(ps.keys()),
                        size=len(dataset),
                        p=list(ps.values()))
                )
                for column, ps in probs.items()
            }
        )

    return df


def create_baselines(dataset_name, data_path, null_prop=None, rounding=2):
    dataset, schema, _ = load_dataset(dataset_name, drop_na=null_prop is None)

    baselines = {
        "baseline_domain": generate_baseline_domain(
            dataset, schema, null_prop=null_prop
        ),
        "baseline_univariate": generate_baseline_univariate(dataset, schema, rounding),
    }

    for name, dataset in baselines.items():
        save_dataset(dataset, data_path, name)


def create_upsampled(dataset_name, other_dataset_name, data_path):
    other_dataset, _, _ = load_dataset(other_dataset_name)
    num_records = len(other_dataset)

    dataset, _, _ = load_dataset(dataset_name)

    upsampled_dataset = dataset.sample(
        num_records, replace=True, random_state=RANDOM_SEED_GENERATION
    )

    _, dataset_core_name = dataset_name.split("/")
    name = f"{dataset_core_name}_upsampled"

    save_dataset(upsampled_dataset, data_path, name)


def create_arbitrary(dataset_name, data_path, max_degree=ARBITRARY_MAX_DEGREE, alpha=ARBITRARY_ALPHA):
    dataset, schema, _ = load_dataset(dataset_name)
    num_records = len(dataset)

    rbn = RandomBayesianNetwork(schema, max_degree, alpha, RANDOM_SEED_GENERATION)
    rbn.print_structure()
    arbitrary_dataset = rbn.sample(num_records)

    save_dataset(arbitrary_dataset, data_path, "arbitrary")


if __name__ == "__main__":

    acs_path = DATA_ROOT / "acs"
    edad_path = DATA_ROOT / "edad"
    we_path = DATA_ROOT / "we"

    create_baselines("acs/national", acs_path)
    create_upsampled("acs/massachusetts", "acs/national", acs_path)
    create_upsampled("acs/texas", "acs/national", acs_path)
    create_arbitrary("acs/national", acs_path)

    create_baselines("edad/2023", edad_path)
    create_arbitrary("edad/2023", edad_path)

    create_baselines("we/2023", we_path)
    create_arbitrary("we/2023", we_path)
