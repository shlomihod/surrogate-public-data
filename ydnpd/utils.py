import os
from typing import Any
from contextlib import contextmanager, redirect_stdout, redirect_stderr

import pandera as pa


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)


def _freeze(d):
    """Recursively freezes a dictionary, converting it and its nested dictionaries to immutable versions."""
    if isinstance(d, dict):
        return tuple((k, _freeze(v)) for k, v in sorted(d.items()))
    elif isinstance(d, list):
        return tuple(_freeze(v) for v in d)
    elif isinstance(d, set):
        return frozenset(_freeze(v) for v in d)
    else:
        return d


def metadata_to_pandera_schema(metadata_schema: dict[str, Any], coerce=False) -> pa.DataFrameSchema:
    schema_dict = {}

    for column_name, column_info in metadata_schema.items():
        dtype = column_info["dtype"]
        values = column_info.get("values")
        checks = []

        if isinstance(values, list):
            checks.append(pa.Check.isin(values))
        elif isinstance(values, dict):
            allowed_values = list(values.keys())
            checks.append(pa.Check.isin(allowed_values))

        schema_dict[column_name] = pa.Column(
            dtype=dtype,
            checks=checks,
            title=column_info.get("description", column_name),
            nullable=False,
        )

    schema = pa.DataFrameSchema(
        schema_dict,
        strict=True,
        coerce=coerce
    )

    return schema


def get_compute_resources():
    import psutil
    import jax
    return {
        "num_cpus": psutil.cpu_count(logical=False),  # physical cores only
        "num_gpus": len(jax.devices("gpu")),
    }
