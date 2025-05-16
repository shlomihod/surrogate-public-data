import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apricot.functions.facilityLocation import FacilityLocationSelection

from ydnpd.datasets import calc_dataset_similarity


MIXING_RANDOM_STATE = 42


def _uniform_dataset_mix(dfs, num_samples, random_state=MIXING_RANDOM_STATE):
    return (pd.concat(dfs)
            .sample(num_samples, replace=False, random_state=random_state)
            .reset_index(drop=True)) 


def mix_datasets(dfs, num_samples, method, num_datasets=None, random_state=MIXING_RANDOM_STATE):

    match method:

        case "uniform":
            if num_datasets is not None:
                raise ValueError("num_datasets must be None with method `uniform`")
            return _uniform_dataset_mix(dfs, num_samples, random_state)

        case "max_coverage":
            datasets = dict(enumerate(dfs))
            similarty_matrix = calc_dataset_similarity(datasets)
            selector = (FacilityLocationSelection(num_datasets, metric="precomputed")
                        .fit(similarty_matrix))
            selected_dfs = [dfs[i] for i in selector.ranking]
            return _uniform_dataset_mix(selected_dfs, num_samples, random_state)

        case _:
            raise ValueError(f"method shoud be either `uniform` or `max_coverage`")


def plot_dataset_selection(dfs, max_num_datasets=None):
    if max_num_datasets is None:
        max_num_datasets = len(dfs)

    datasets = dict(enumerate(dfs))
    similarity_matrix = calc_dataset_similarity(datasets)

    objectives = []
    selected_sets = []

    # Calculate objectives as before
    for k in range(1, max_num_datasets + 1):
        selector = (FacilityLocationSelection(k, metric="precomputed")
                    .fit(similarity_matrix))
        selected_sets.append(selector.ranking)
        objectives.append(selector.gains.sum())

    # Calculate rates of change
    rates_of_change = np.diff(objectives)
    relative_improvements = rates_of_change / objectives[:-1] * 100

    # Find points where relative improvement drops below thresholds
    threshold_indices = {
        '5%': next((i for i, imp in enumerate(relative_improvements) if imp < 5), None),
        '1%': next((i for i, imp in enumerate(relative_improvements) if imp < 1), None)
    }

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Original coverage plot
    ax1.plot(range(1, max_num_datasets + 1), objectives, 'bo-')
    ax1.set_xlabel('Number of datasets')
    ax1.set_ylabel('Coverage Objective')
    ax1.grid(True)

    # Rate of change plot
    ax2.plot(range(2, max_num_datasets + 1), relative_improvements, 'ro-')
    ax2.set_xlabel('Number of datasets')
    ax2.set_ylabel('Relative Improvement (%)')
    ax2.grid(True)

    plt.tight_layout()

    return fig, {
        'objectives': objectives,
        'selected_sets': selected_sets,
        'rates_of_change': rates_of_change.tolist(),
        'relative_improvements': relative_improvements.tolist(),
        'threshold_points': threshold_indices
    }
