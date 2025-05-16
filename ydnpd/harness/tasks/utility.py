import itertools as it

import traceback

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import plotly.graph_objects as go
import plotly.subplots as sp
import wandb

from ydnpd.utils import _freeze
from ydnpd.datasets.loader import load_dataset, split_train_eval_datasets
from ydnpd.harness.synthesis import generate_synthetic_data
from ydnpd.harness.evaluation import evaluate_two, EVALUATION_METRICS
from ydnpd.harness.tasks import DPTask


PLOT_RADAR_JITTER = 0.005


# plt.style.use(['science', 'no-latex'])


class UtilityTask(DPTask):

    METRIC_DEFAULT = "marginals_3_max_abs_diff_error"

    def __init__(
        self,
        dataset_pointer: str | tuple[str, str],
        epsilons: list[float],
        synth_name: str,
        hparam_dims: dict[str, list],
        num_runs: int,
        verbose: bool = True,
        with_wandb: bool = False,
        wandb_kwargs: dict = None,
        evaluation_kwargs: dict = None,
    ):

        self.dataset_pointer = dataset_pointer
        if isinstance(dataset_pointer, str):
            self.dataset_name, self.dataset_path = dataset_pointer, None
        elif isinstance(dataset_pointer, tuple):
            self.dataset_name, self.dataset_path = dataset_pointer
        else:
            raise TypeError(f"`dataset_pointer` should be either string or 2-tuple of strings")

        self.dataset_family, _ = self.dataset_name.split("/")

        self.epsilons = epsilons
        self.synth_name = synth_name
        self.hparam_dims = hparam_dims
        self.num_runs = num_runs
        self.verbose = verbose
        self.with_wandb = with_wandb
        self.evaluation_kwargs = (
            evaluation_kwargs["_"] | evaluation_kwargs[self.dataset_family]
            if evaluation_kwargs is not None else {}
        )

        if not with_wandb:
            if wandb_kwargs is not None:
                raise ValueError("`wandb_kwargs` must be None if `with_wandb` is False")
        else:
            self.wandb_kwargs = wandb_kwargs if wandb_kwargs is not None else {}

        self.hparam_space = [
            dict(zip(hparam_dims, values))
            for values in it.product(*hparam_dims.values())
        ]

    def __str__(self):
        return f"<UtilityTask (#configs={self.size()}): {self.synth_name} & {self.dataset_name}>"

    def size(self) -> int:
        return len(self.epsilons) * len(self.hparam_space) * self.num_runs

    def _execute_run(
        self,
        train_dataset: pd.DataFrame,
        eval_dataset: pd.DataFrame,
        schema: dict,
        epsilon: float,
        hparams: dict,
    ) -> tuple[dict, pd.DataFrame]:
        synth_dataset = generate_synthetic_data(
            train_dataset, schema, epsilon, self.synth_name, **hparams
        )
        # NOTE: this is important for GEM, which returns an aligned
        # dataset with the train dataset, but with renamed columns
        synth_dataset.columns = train_dataset.columns

        metric_results = evaluate_two(
            train_dataset, eval_dataset, synth_dataset, schema, **self.evaluation_kwargs
        )

        return metric_results, synth_dataset

    def execute(self) -> list[dict]:

        dataset, schema, _ = load_dataset(self.dataset_name, self.dataset_path)

        train_dataset, eval_dataset = split_train_eval_datasets(dataset)

        results = []

        for epsilon in self.epsilons:

            for hparams in self.hparam_space:

                config = {
                    "dataset_name": self.dataset_name,
                    "epsilon": epsilon,
                    "synth_name": self.synth_name,
                    "hparams": hparams,
                }

                if self.with_wandb:
                    wandb.init(project="ydnpd-harness", config=config, **self.wandb_kwargs)

                for run_id in range(self.num_runs):

                    if self.verbose:
                        print(
                            f"{self.__class__.__name__}: dataset = {self.dataset_name} synth_name={self.synth_name}, epsilon={epsilon}, hparams={hparams} run={run_id + 1}/{self.num_runs}"
                        )

                    try:
                        (metric_results, synth_dataset) = self._execute_run(
                            train_dataset, eval_dataset, schema, epsilon, hparams
                        )
                    except Exception as e:
                        print(traceback.format_exc())

                        if self.with_wandb:
                            wandb.log({"_error": str(e)})

                        metric_results = {}
                        synth_dataset = None
                    else:
                        if self.with_wandb:
                            wandb.log({key: value if not isinstance(value, np.ndarray) else value.tolist()
                                       for key, value in metric_results.items()})

                    results.append(
                        config
                        | {"evaluation": metric_results, "synth_dataset": synth_dataset}
                    )

                if self.with_wandb:
                    wandb.finish()

        return results

    # @staticmethod
    # def evaluate(hparam_results, experiments, metric=None):
    #     if metric is None:
    #         metric = UtilityTask.METRIC_DEFAULT
    #     metric_column = f"evaluation_{metric}"

    #     results_df = UtilityTask.process(hparam_results, experiments)

    #     best_hparams_df = results_df.groupby(["dataset_name", "synth_name", "epsilon"])[
    #         metric_column
    #     ].idxmin()

    #     def extractor(test_name, dev_name):

    #         def function(r):
    #             metric_column = f"evaluation_{metric}"
    #             synth_name, epsilon = r.name
    #             hparams = results_df.iloc[r.item()]["hparams_frozen"]
    #             base_mask = (results_df["synth_name"] == synth_name) & (
    #                 results_df["epsilon"] == epsilon
    #             )
    #             best_dev_result = results_df.loc[
    #                 (results_df["dataset_name"] == dev_name)
    #                 & (results_df["hparams_frozen"] == hparams)
    #                 & base_mask,
    #                 metric_column,
    #             ].item()

    #             correspond_test_result = results_df.loc[
    #                 (results_df["dataset_name"] == test_name)
    #                 & (results_df["hparams_frozen"] == hparams)
    #                 & base_mask,
    #                 metric_column,
    #             ].item()
    #             test_results = results_df.loc[
    #                 (results_df["dataset_name"] == test_name) & base_mask, metric_column
    #             ]
    #             return {
    #                 "quantile": (correspond_test_result > test_results).sum()
    #                 / len(test_results),
    #                 "best_dev": best_dev_result,
    #                 "correspond_test": correspond_test_result,
    #                 "best_test": test_results.min(),
    #                 "median_test": test_results.median(),
    #                 "worst_test": test_results.max(),
    #             }

    #         return function

    #     hparams_evaluation_df = (
    #         pd.concat(
    #             [
    #                 (
    #                     pd.DataFrame(best_hparams_df[dev_name])
    #                     .apply(
    #                         extractor(experiments.test_name, dev_name),
    #                         result_type="expand",
    #                         axis=1,
    #                     )
    #                     .reset_index()
    #                     .assign(
    #                         dev_name=dev_name,
    #                         test_name=experiments.test_name,
    #                         experiment=f"{experiments.test_name}/{dev_name}",
    #                     )
    #                 )
    #                 for dev_name in experiments.dev_names
    #             ]
    #         )
    #         .set_index(["synth_name", "experiment", "epsilon"])
    #         .sort_index()
    #         .drop(columns=["test_name", "dev_name"])
    #     )
            

    #     return hparams_evaluation_df

    @staticmethod
    def evaluate(hparam_results, experiments, metric=None):
        if metric is None:
            metric = UtilityTask.METRIC_DEFAULT
        metric_column = f"evaluation_{metric}"

        results_df = UtilityTask.process(hparam_results, experiments)

        best_hparams_df = results_df.groupby(["dataset_name", "synth_name", "epsilon"])[
            metric_column
        ].idxmin()

        def extractor(test_name, dev_name):

            def function(r):
                metric_column = f"evaluation_{metric}"
                synth_name, epsilon = r.name

                try:
                    # Extract the frozen hyperparameters for the current configuration
                    hparams = results_df.iloc[r.item()]["hparams_frozen"]

                    # Base mask to filter results_df
                    base_mask = (results_df["synth_name"] == synth_name) & (
                        results_df["epsilon"] == epsilon
                    )

                    # Fetch best dev result
                    best_dev_result = results_df.loc[
                        (results_df["dataset_name"] == dev_name)
                        & (results_df["hparams_frozen"] == hparams)
                        & base_mask,
                        metric_column,
                    ]

                    # Fetch corresponding test result
                    correspond_test_result = results_df.loc[
                        (results_df["dataset_name"] == test_name)
                        & (results_df["hparams_frozen"] == hparams)
                        & base_mask,
                        metric_column,
                    ]

                    # Fetch all test results for quantile computation
                    test_results = results_df.loc[
                        (results_df["dataset_name"] == test_name) & base_mask,
                        metric_column,
                    ]

                    # Ensure results are valid
                    if best_dev_result.size != 1 or correspond_test_result.size != 1:
                        raise ValueError(
                            f"Unexpected number of results: best_dev_result.size={best_dev_result.size}, "
                            f"correspond_test_result.size={correspond_test_result.size}.\n"
                            f"Details:\n"
                            # f" - synth_name={synth_name}\n"
                            # f" - epsilon={epsilon}\n"
                            # f" - dev_name={dev_name}\n"
                            # f" - test_name={test_name}\n"
                            # f" - hparams={hparams}\n"
                            # f" - best_dev_result:\n{best_dev_result}\n"
                            # f" - correspond_test_result:\n{correspond_test_result}"
                        )
                    

                    return {
                        "quantile": (correspond_test_result.item() > test_results).sum()
                        / len(test_results),
                        "best_dev": best_dev_result.item(),
                        "correspond_test": correspond_test_result.item(),
                        "best_test": test_results.min(),
                        "median_test": test_results.median(),
                        "worst_test": test_results.max(),
                    }

                except Exception as e:
                    print(
                        f"Error in extractor function:\n"
                        f" - synth_name: {synth_name}\n"
                        f" - epsilon: {epsilon}\n"
                        f" - dev_name: {dev_name}\n"
                        f" - test_name: {test_name}\n"
                        # f" - hparams: {hparams}\n"
                        # f" - Results DataFrame snippet:\n{results_df[0]}\n"
                        f" - Error: {e}"
                    )
                    raise

            return function


        try:
            hparams_evaluation_df = (
                pd.concat(
                    [
                        (
                            pd.DataFrame(best_hparams_df[dev_name])
                            .apply(
                                extractor(experiments.test_name, dev_name),
                                result_type="expand",
                                axis=1,
                            )
                            .reset_index()
                            .assign(
                                dev_name=dev_name,
                                test_name=experiments.test_name,
                                experiment=f"{experiments.test_name}/{dev_name}",
                            )
                        )
                        for dev_name in experiments.dev_names
                    ]
                )
                .set_index(["synth_name", "experiment", "epsilon"])
                .sort_index()
                .drop(columns=["test_name", "dev_name"])
            )
        except Exception as e:
            print(f"Error in evaluate method: {e}")
            # print(f"Results DataFrame:\n{results_df.head()}")
            raise

        return hparams_evaluation_df


    @staticmethod
    def plot(hparam_results, experiments, metric=None):
        sns.set_context("paper", rc={"axes.titlesize":16,
                                     "axes.labelsize":14,
                                     "xtick.labelsize":12,
                                     "ytick.labelsize":12})

        if metric is None:
            metric = UtilityTask.METRIC_DEFAULT
        metric_column = f"evaluation_{metric}"

        results_df = UtilityTask.process(hparam_results, experiments)

        evaluation_df = UtilityTask.evaluate(
            hparam_results, experiments, metric
        ).reset_index()

        results_df[metric_column] *= 100
        value_columns = ["best_dev", "best_test", "correspond_test"]
        evaluation_df[value_columns] *= 100

        epsilons = results_df["epsilon"].unique()

        def plot_dev_vs_test():
            def expender(test_name, dev_name):
                def function(g):
                    g = g[["dataset_name", "hparams_frozen", metric_column]].rename(
                        columns={metric_column: "metric"}
                    )
                    df = (
                        pd.merge(
                            g.query("dataset_name == @test_name"),
                            g.query("dataset_name == @dev_name"),
                            on="hparams_frozen",
                            suffixes=("_test", "_dev"),
                        )
                        .rename(
                            columns={
                                "dataset_name_test": "test_name",
                                "dataset_name_dev": "dev_name",
                            }
                        )
                        .assign(
                            experiment=f"{test_name}/{dev_name}",
                        )
                    )
                    return df

                return function

            experiment_df = pd.concat(
                [
                    results_df.groupby(["synth_name", "epsilon"])
                    .apply(
                        expender(experiments.test_name, dev_name), include_groups=False
                    )
                    .reset_index()
                    for dev_name in experiments.dev_names
                ]
            )

            g = sns.lmplot(
                data=experiment_df,
                x="metric_dev",
                y="metric_test",
                hue="epsilon",
                row="synth_name",
                col="experiment",
                ci=None,
                height=4,
                aspect=1,
            )

            g.set(xlim=(0, 100), ylim=(0, 100))
            g.set_titles("{row_name} | {col_name}")
            g.set_axis_labels("Dev Metric (%)", "Test Metric (%)")
            g.fig.suptitle(f"Dev (per panel) vs Test ({experiments.test_name}) Performance for Each HParam Configuration", y=1.02, fontsize=18)

            return g

        def plot_dev_within_test():
            def plot_swarm_and_line(data, **kwargs):

                synth_name = data["synth_name"].unique().item()

                sns.swarmplot(
                    data=data, x="epsilon", y=metric_column, color="black", alpha=0.6
                )

                sns.pointplot(
                    data=evaluation_df.query(f"synth_name == '{synth_name}'"),
                    x="epsilon",
                    y="correspond_test",
                    hue="experiment",
                    marker="x",
                )

            g = sns.FacetGrid(
                results_df[results_df["dataset_name"] == experiments.test_name],
                col="synth_name",
                height=4,
                aspect=1,
            )
            g.map_dataframe(plot_swarm_and_line)

            g.add_legend()
            g.set_axis_labels(r"$\epsilon$", f"{metric} (%)")
            g.set_titles("{col_name}")
            g.fig.suptitle(f"Test ({experiments.test_name}) Performance If Best HParams Choosen According to Dev (per line)", y=1.02, fontsize=18)


            return g

        def plot_best_dev():

            best_hparams_df = (
                results_df.groupby(["dataset_name", "synth_name", "epsilon"])[
                    metric_column
                ]
                .mean()
                .reset_index()
            )

            g = sns.relplot(
                data=best_hparams_df,
                x="epsilon",
                y=metric_column,
                hue="dataset_name",
                col="synth_name",
                kind="line",
                errorbar=None,
                height=4,
                aspect=1,
            )

            g.set(xticks=epsilons)
            g.set_axis_labels(r"$\epsilon$", f"Mean {metric} (%)")
            g.set_titles("{col_name}")
            g.fig.suptitle(f"Privacy-Utility Tradeoff: Dev Performance If Best HParams Choosen According to Dev (per line)", y=1.02, fontsize=18)


            return g

        def plot_best_dev_vs_test():

            melted_df = evaluation_df.melt(
                id_vars=["synth_name", "experiment", "epsilon"],
                value_vars=["best_dev", "best_test", "correspond_test"],
                var_name="metric",
                value_name="value",
            )

            g = sns.relplot(
                data=melted_df,
                x="epsilon",
                y="value",
                hue="metric",
                col="experiment",
                row="synth_name",
                kind="line",
                errorbar=None,
                height=4,
                aspect=1,
            )

            g.set(xticks=epsilons)
            g.set_axis_labels(r"$\epsilon$", "Metric Value (%)")
            g.set_titles("{row_name} | {col_name}")
            g.fig.suptitle(f"Comparing Performacece of (1) Best Dev by Dev; (2) Best Test by Test; (3) Best Test by Dev", y=1.02, fontsize=18)

            return g

        return (
            plot_dev_vs_test(),
            plot_dev_within_test(),
            plot_best_dev(),
            plot_best_dev_vs_test(),
        )

    def plot_overall(hparam_results, experiments, epsilon_reference):
        # Get core evaluation metrics
        core_evaluation_metrics = [metric
                                for metric in EVALUATION_METRICS
                                if "train_dataset" not in metric]

        # Create evaluation dataframe
        all_evaluation_df = pd.concat(
            [
                UtilityTask.evaluate(
                    hparam_results,
                    experiments,
                    metric).assign(metric=metric)
            for metric in core_evaluation_metrics
            ]).reset_index()

        # Filter for reference epsilon
        ref_eps_all_evaluation_df = all_evaluation_df[all_evaluation_df["epsilon"] == epsilon_reference]

        # Create color mapping
        dataset_members = sorted(set([exp.split('/')[-1] for exp in ref_eps_all_evaluation_df['experiment'].unique()]))
        tab10 = plt.cm.tab10(np.linspace(0, 1, len(dataset_members)))
        color_dict = dict(zip(dataset_members, [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                                            for r, g, b, _ in tab10[:len(dataset_members)]]))

        # Create matplotlib color dict for seaborn plots
        mpl_color_dict = dict(zip(dataset_members, [tab10[i] for i in range(len(dataset_members))]))

        # Initialize figures list
        figs = []

        for measure in ["correspond_test", "best_dev"]:
            print(f"\nProcessing measure: {measure}")

            # Process data for radar plot - modified to handle the structure correctly
            result_traces = {}
            for synth_name in ref_eps_all_evaluation_df['synth_name'].unique():
                synth_data = ref_eps_all_evaluation_df[ref_eps_all_evaluation_df['synth_name'] == synth_name]
                experiment_data = {}

                for experiment in synth_data['experiment'].unique():
                    data = synth_data[synth_data['experiment'] == experiment][measure].to_numpy()
                    dataset_member = experiment.split('/')[-1]
                    experiment_data[dataset_member] = data

                result_traces[synth_name] = experiment_data

            # Create separate radar plots for each synth_name
            for synth_name, synth_results in result_traces.items():
                radar_fig = go.Figure()

                legend_entries = set()
                for dataset_member, trace in synth_results.items():
                    jitter = np.random.normal(0, PLOT_RADAR_JITTER, size=len(trace))
                    jittered_trace = trace + jitter

                    r_values = np.append(jittered_trace, jittered_trace[0])
                    theta_values = np.append(core_evaluation_metrics, core_evaluation_metrics[0])

                    showlegend = dataset_member not in legend_entries
                    if showlegend:
                        legend_entries.add(dataset_member)

                    radar_fig.add_trace(
                        go.Scatterpolar(
                            r=r_values,
                            theta=theta_values,
                            opacity=1,
                            name=dataset_member,
                            showlegend=showlegend,
                            line=dict(color=color_dict[dataset_member])
                        )
                    )

                # Update layout for this specific radar plot
                max_value = np.ceil(ref_eps_all_evaluation_df[measure].max() * 20) / 20
                min_value = np.floor(ref_eps_all_evaluation_df[measure].min() * 20) / 20

                radar_fig.update_layout(
                    title=f"{measure} - {synth_name}",
                    polar=dict(
                        radialaxis=dict(visible=True, range=[min_value, max_value]),
                        angularaxis=dict(tickfont=dict(size=8))
                    ),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.1,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=10),
                        itemwidth=40
                    ),
                    height=500,
                )

                figs.append(radar_fig)

            # Create bar plot using seaborn
            class SeabornFigure:
                def __init__(self, data, measure, synth_names, color_dict, core_metrics, max_value, min_value):
                    self.data = data
                    self.measure = measure
                    self.synth_names = synth_names
                    self.color_dict = color_dict
                    self.core_metrics = core_metrics
                    self.max_value = max_value
                    self.min_value = min_value

                def show(self):
                    fig, axes = plt.subplots(len(self.synth_names), 1,
                                        figsize=(20, 6*len(self.synth_names)),
                                        squeeze=False)
                    axes = axes.flatten()

                    for idx, synth_name in enumerate(self.synth_names):
                        synth_data = self.data[self.data['synth_name'] == synth_name]

                        plot_data = []
                        for metric in self.core_metrics:
                            metric_data = synth_data[synth_data['metric'] == metric]
                            for exp in metric_data['experiment'].unique():
                                dataset_member = exp.split('/')[-1]
                                plot_data.append({
                                    'Metric': metric,
                                    'Value': metric_data[metric_data['experiment'] == exp][self.measure].iloc[0],
                                    'Dataset': dataset_member
                                })

                        plot_df = pd.DataFrame(plot_data)

                        sns.barplot(data=plot_df,
                                x='Metric',
                                y='Value',
                                hue='Dataset',
                                palette=self.color_dict,
                                ax=axes[idx],
                                width=0.9)

                        axes[idx].set_title(f"{self.measure} - {synth_name}")
                        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=90, ha='right', va='center')
                        axes[idx].tick_params(axis='x', labelrotation=90, pad=150)
                        axes[idx].set_ylim(self.min_value, self.max_value)
                        axes[idx].margins(x=0.05)

                        if idx != 0:
                            axes[idx].get_legend().remove()

                    plt.legend(bbox_to_anchor=(0.5, len(self.synth_names) + 0.2),
                            loc='center',
                            ncol=len(dataset_members),
                            borderaxespad=0.)

                    plt.subplots_adjust(wspace=0.2)
                    plt.tight_layout(rect=[0, 2, 1, 0.95])
                    display(fig)
                    plt.close()

            # Create seaborn figure
            seaborn_fig = SeabornFigure(
                data=ref_eps_all_evaluation_df,
                measure=measure,
                synth_names=ref_eps_all_evaluation_df['synth_name'].unique(),
                color_dict=mpl_color_dict,
                core_metrics=core_evaluation_metrics,
                max_value=np.ceil(ref_eps_all_evaluation_df[measure].max() * 20) / 20,
                min_value=np.floor(ref_eps_all_evaluation_df[measure].min() * 20) / 20
            )

            figs.append(seaborn_fig)

        return figs

    @staticmethod
    def process(hparam_results, experiments):

        df = pd.DataFrame(hparam_results)

        dataset_names = list(set([experiments.test_name] + experiments.dev_names))
        df = df[df["dataset_name"].isin(dataset_names)].reset_index(drop=True)

        df["hparams_frozen"] = df["hparams"].apply(_freeze)

        metric_columns = []
        for metric in EVALUATION_METRICS:
            if not metric.startswith("_"):
                metric_column = f"evaluation_{metric}"
                df[metric_column] = df["evaluation"].apply(lambda x: x[metric])
                metric_columns.append(metric_column)

        df = (
            df.groupby(["dataset_name", "synth_name", "epsilon", "hparams_frozen"])[
                metric_columns
            ]
            .mean()
            .reset_index()
        )

        return df
