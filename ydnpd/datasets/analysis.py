import itertools as it
from typing import Any

import numpy as np
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from ydata_profiling import ProfileReport, compare

from ydnpd.harness.evaluation import calc_k_marginals_abs_diff_errors

DEFAULT_DISTANCE_METRIC = "total_variation"

def calc_dataset_similarity(datasets: dict[Any, pd.DataFrame], metric: str = DEFAULT_DISTANCE_METRIC):
    probs = [df.value_counts(normalize=True) for df in datasets.values()]

    probs_df = (
        pd.concat(probs, axis=1,
                  keys=datasets.keys())
                  .fillna(0)
                  )

    match metric:
        case "total_variation":
            dist_matrix = cdist(probs_df.T, probs_df.T, metric='cityblock') / 2
        case "jensenshannon":
            dist_matrix = cdist(probs_df.T, probs_df.T, metric='jensenshannon')
        case "3_way_marginals":
            # creata a distance matrix for 3-way marginals
            dist_matrix = np.zeros((len(datasets), len(datasets)))
            for (idx1, df1), (idx2, df2) in it.combinations(enumerate(datasets.values()), 2):
                dist_matrix[idx1, idx2] = calc_k_marginals_abs_diff_errors(df1, df2, marginals_k=3)["marginals_3_avg_abs_diff_error"]
                dist_matrix[idx2, idx1] = dist_matrix[idx1, idx2]
        case _:
            raise ValueError(f"metric {metric} is unkonwn.")

    similarity_matrix = (1 - dist_matrix).clip(0, 1)
    similarity_df = pd.DataFrame(similarity_matrix, columns=datasets.keys(), index=datasets.keys())

    return similarity_df


def plot_distribution_distances(
    datasets: dict[Any, pd.DataFrame],
    metric: str = DEFAULT_DISTANCE_METRIC,
    with_clustering: bool = True,
    # figsize: tuple[int, int] = (20, 8)
) -> None:
    """
    Create visualizations of distribution distances using clustered heatmap and 2D TSNE.

    Args:
        datasets: Dictionary mapping dataset identifiers to DataFrames containing distributions
        metric: Name of the distance metric used (for plot labels)
        figsize: Size of the figure as (width, height)
    """
    # Compute distance matrix

    dist_matrix = (calc_dataset_similarity(datasets, metric))
    labels = list(datasets.keys())

    # 1. Clustered Heatmap
    g = sns.clustermap(
        dist_matrix.iloc[:-1, 1:].multiply(100).round(1),
        row_cluster=with_clustering,
        col_cluster=with_clustering,
        mask=(np.tril(np.ones_like(dist_matrix, dtype=bool))[:-1, 1:]),
        cmap="magma",
        vmin=0,
        vmax=100,
        annot=True,
        fmt="g",
        annot_kws={'size': 12},
        xticklabels=labels[1:],
        yticklabels=labels[:-1],
        figsize=(10, 10),
        dendrogram_ratio=0.1,
        
        cbar_pos=(0.8, 0.3, 0.03, 0.5),
    )

    g.ax_heatmap.tick_params(axis='x', labelsize=12) 
    g.ax_heatmap.tick_params(axis='y', labelsize=12)

    # Rotate axis labels
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), 
                             rotation=45,
                             ha='left',
                             rotation_mode='anchor')
    g.ax_heatmap.tick_params(axis='x', pad=10)

    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    # # Move x-axis labels to the top
    g.ax_heatmap.xaxis.tick_top()
    g.ax_heatmap.xaxis.set_label_position('top')

    # # Move y-axis labels to the right
    g.ax_heatmap.yaxis.tick_left()
    g.ax_heatmap.yaxis.set_label_position('left')

    return g


    # 2. 2D TSNE
    # coords_2d = TSNE(
    #     n_components=2,
    #     metric='precomputed',
    #     init='random',
    #     random_state=42
    # ).fit_transform(dist_matrix)

    # plt.figure(figsize=(10, 8))
    # sns.scatterplot(
    #     x=coords_2d[:, 0],
    #     y=coords_2d[:, 1],
    #     hue=base_labels,
    #     s=100,
    #     alpha=0.8,
    # )

    # # Add labels to points
    # for i, label in enumerate(labels):
    #     plt.annotate(
    #         str(label),
    #         (coords_2d[i, 0], coords_2d[i, 1]),
    #         xytext=(5, 5),
    #         textcoords='offset points',
    #         bbox=dict(
    #             facecolor='white',
    #             edgecolor='none',
    #             alpha=0.1,
    #             pad=1
    #         )
    #     )

    # plt.title(f'2D TSNE Visualization of {metric.title()} Distances')
    # plt.xlabel('First dimension')
    # plt.ylabel('Second dimension')
    # plt.legend(title="Groups", bbox_to_anchor=(1.05, 1), loc='upper left')
    # sns.despine()
    # plt.tight_layout()
    # plt.show()


def compare_datasets(datasets: dict[str, pd.DataFrame]) -> Any:
    """
    Generate comparison ProfileReport for a dictionary of datasets.

    Args:
        datasets: Dictionary mapping dataset names to their DataFrames

    Returns:
        comparison: The ydata_profiling comparison object
    """
    reports = [
        ProfileReport(df, title=name) 
        for name, df in datasets.items()
    ]

    return compare(reports)
