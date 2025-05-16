import json
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from ydnpd.utils import metadata_to_pandera_schema


RANDOM_STATE_TRAIN_TEST_SPLIT = 42
EVAL_SPLIT_PROPORTION = 0.3

DATA_ROOT = Path(__file__).parent / "data"

# https://pages.nist.gov/privacy_collaborative_research_cycle/pages/participate.html
# maybe only run on the categorical columns>?
COL_SUBSETS = {
    "acs": {
        "demographic": [
            "SEX",
            "MSP",
            "RAC1P",
            "OWN_RENT",
            "PINCP_DECILE",
            "EDU",
            "HOUSING_TYPE",
            # "DVET",  # Many missing values
            # "DEYE",
            # "AGEP",  # Continuous
        ]
    }
}


def load_dataset(
    dataset_name: str, path: Optional[str] = None,
    *, cols_subset_name: str | bool = True, drop_na: bool = True
):

    family, member = dataset_name.split("/")
    metadata_path = DATA_ROOT / family / "metadata.json"

    if not metadata_path.exists():
        raise ValueError(f"dataset family `{family}` unkown")

    dataset_dir_path = DATA_ROOT if path is None else Path(path)

    dataset_path = dataset_dir_path / family / f"{member}.csv"
    if not dataset_path.exists():
        raise ValueError(f"dataset member `{member}` is not part of family `{family}` in path `{dataset_dir_path}`")

    dataset = pd.read_csv(dataset_path)
    metadata = json.load(open(metadata_path))
    schema = metadata["schema"]
    domain = metadata["domain"]

    if cols_subset_name:
        if family in COL_SUBSETS:
            if family == "acs":
                cols_subset_name = "demographic"

            col_subset = COL_SUBSETS[family][cols_subset_name]
            dataset = dataset[col_subset]
            schema = {k: v for k, v in schema.items() if k in col_subset}

    processed_schema = {}

    for col in dataset.columns:
        col_schema = schema[col].copy()

        if drop_na:
            if col_schema.pop("has_null", False):
                null_value = col_schema.pop("null_value")
                dataset = dataset[dataset[col] != null_value]
                col_schema["values"] = {
                    k: v for k, v in col_schema["values"].items() if k != null_value
                }
                col_schema["dtype"] = "int64"

        if col_schema["dtype"].startswith("int"):
            col_schema["values"] = {
                int(k): v for k, v in col_schema["values"].items()
            }

        processed_schema[col] = col_schema
        dataset[col] = dataset[col].astype(processed_schema[col]["dtype"])

    pa_schema = metadata_to_pandera_schema(processed_schema)
    pa_schema.validate(dataset)

    return dataset, processed_schema, domain


def split_train_eval_datasets(dataset):

    return train_test_split(
        dataset,
        test_size=EVAL_SPLIT_PROPORTION,
        random_state=RANDOM_STATE_TRAIN_TEST_SPLIT,
    )
