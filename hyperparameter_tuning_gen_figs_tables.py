import os
import glob
import pickle
import time
import json

os.environ["PRIVBAYES_BIN"] = "./ydnpd/harness/synthesis/privbayes/mac_bin"

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display, Markdown
import seaborn as sns

import ydnpd
from additional import ADDITIONAL_EXPERIMENTS, ADDITIONAL_PATH
from ydnpd import EVALUATION_METRICS, ALL_EXPERIMENTS, Experiments

from table_utils import df_to_latex_medals, create_latex_table_with_medals

ADDITIONAL_DATASETS = sum(list(ADDITIONAL_EXPERIMENTS.values()), [])

with open("./results/harness-gen.pkl", "rb") as f:
    utility_tasks_results_gem = pickle.load(f)

with open("./results/harness-multi.pkl", "rb") as f:
    utility_tasks_results_multi = pickle.load(f)

utility_tasks_results = utility_tasks_results_multi + utility_tasks_results_gem

for x in utility_tasks_results:
    if type(x["evaluation"]["error_rate_diff"]) == list:
        assert len(x["evaluation"]["error_rate_diff"]) == 1
        x["evaluation"]["error_rate_diff"] = x["evaluation"]["error_rate_diff"][0]

utility_tasks_results = [x for x in utility_tasks_results if "sdscm" not in x["dataset_name"].lower()]

missing_metrics = {}
for x in utility_tasks_results:
    for metric in EVALUATION_METRICS:
        if np.isnan(x["evaluation"][metric]):
            missing_metrics[x["dataset_name"]] = missing_metrics.get(x["dataset_name"], []) + [metric]
all_results = utility_tasks_results.copy()

reference_data_mapping = {
    "acs": "acs/national",
    "edad": "edad/2023",
    "we": "we/2023",
}

METRIC_DIRECTION = {
    "error_rate_diff": "closer_to_zero_is_better",
    "auc_diff": "closer_to_zero_is_better",
    "marginals_3_max_abs_diff_error": "closer_to_zero_is_better",
    "marginals_3_avg_abs_diff_error": "closer_to_zero_is_better",
    "thresholded_marginals_3_max_abs_diff_error": "closer_to_zero_is_better",
    "thresholded_marginals_3_avg_abs_diff_error": "closer_to_zero_is_better",
    "total_variation_distance": "closer_to_zero_is_better",
    "pearson_corr_max_abs_diff": "closer_to_zero_is_better",
    "pearson_corr_avg_abs_diff": "closer_to_zero_is_better",
    "cramer_v_corr_max_abs_diff": "closer_to_zero_is_better",
    "cramer_v_corr_avg_abs_diff": "closer_to_zero_is_better",
}
    
CORRELATION_METRICS = [
    "total_variation_distance",
    "pearson_corr_max_abs_diff",
    "pearson_corr_avg_abs_diff",
    "cramer_v_corr_max_abs_diff",
    "cramer_v_corr_avg_abs_diff",
]
MARGINALS_METRICS = [
    "marginals_3_max_abs_diff_error",
    "marginals_3_avg_abs_diff_error",
    "thresholded_marginals_3_max_abs_diff_error",
    "thresholded_marginals_3_avg_abs_diff_error",
]
CLASSIFICATION_METRICS = [
    "error_rate_diff",
    "auc_diff",
]


metric_group_map = {
    "correlation_metrics": "Correlation",
    "classification_metrics": "Classification",
    "marginals_metrics": "Marginals"
}

we_method_name_map = {
    "we/2018": "Public",
    "we/2023": "Private",
    "we/arbitrary": "Arbitrary",
    "we/baseline_domain": "Domain (Baseline)",
    "we/baseline_univariate": "Univariate (Baseline)",
    "we/csv-claude": "CSV (Claude 3.5 Sonnet)",
    "we/csv-gpt": "CSV (GPT-4o)",
    "we/csv-llama": "CSV (Llama 3.3 70B)",
    "we/gen-MIX-MAX": "Agent (All, Max Cov.)",
    "we/gen-MIX-UNIF": "Agent (All, Unif.)",
    "we/gen-claude-MIX-MAX": "Agent (Claude 3.5 Sonnet, Max Cov.)",
    "we/gen-claude-MIX-UNIF": "Agent (Claude 3.5 Sonnet, Unif.)",
    "we/gen-gpt-MIX-MAX": "Agent (GPT-4o, Max Cov.)",
    "we/gen-gpt-MIX-UNIF": "Agent (GPT-4o, Unif.)",
    "we/gen-llama-MIX-MAX": "Agent (Llama 3.3 70B, Max Cov.)",
    "we/gen-llama-MIX-UNIF": "Agent (Llama 3.3 70B, Unif.)",
    "we/sdscm-gpt2": "SDSCM (GPT-2)",
    "we/sdscm-llama-3-8b": "SDSCM (LLaMA-3 8B)",
    "we/sdscm-olmo-1b-hf": "SDSCM (OLMO-1B HF)"
}

acs_method_name_map = {
    "acs/arbitrary": "Arbitrary",
    "acs/baseline_domain": "Domain (Baseline)",
    "acs/baseline_univariate": "Univariate (Baseline)",
    "acs/csv-claude": "CSV (Claude 3.5 Sonnet)",
    "acs/csv-gpt": "CSV (GPT-4o)",
    "acs/csv-llama": "CSV (Llama 3.3 70B)",
    "acs/gen-MIX-MAX": "Agent (All, Max Cov.)",
    "acs/gen-MIX-UNIF": "Agent (All, Unif.)",
    "acs/gen-claude-MIX-MAX": "Agent (Claude 3.5 Sonnet, Max Cov.)",
    "acs/gen-claude-MIX-UNIF": "Agent (Claude 3.5 Sonnet, Unif.)",
    "acs/gen-gpt-MIX-MAX": "Agent (GPT-4o, Max Cov.)",
    "acs/gen-gpt-MIX-UNIF": "Agent (GPT-4o, Unif.)",
    "acs/gen-llama-MIX-MAX": "Agent (Llama 3.3 70B, Max Cov.)",
    "acs/gen-llama-MIX-UNIF": "Agent (Llama 3.3 70B, Unif.)",
    "acs/massachusetts_upsampled": "Public",
    "acs/national": "Private",
    "acs/sdscm-gpt2": "SDSCM (GPT-2)",
    "acs/sdscm-llama-3-8b": "SDSCM (LLaMA-3 8B)",
    "acs/sdscm-olmo-1b-hf": "SDSCM (OLMO-1B HF)"
}

edad_method_name_map = {
    "edad/2020": "Public",
    "edad/2023": "Private",
    "edad/arbitrary": "Arbitrary",
    "edad/baseline_domain": "Domain (Baseline)",
    "edad/baseline_univariate": "Univariate (Baseline)",
    "edad/csv-claude": "CSV (Claude 3.5 Sonnet)",
    "edad/csv-gpt": "CSV (GPT-4o)",
    "edad/csv-llama": "CSV (Llama 3.3 70B)",
    "edad/gen-MIX-MAX": "Agent (All, Max Cov.)",
    "edad/gen-MIX-UNIF": "Agent (All, Unif.)",
    "edad/gen-claude-MIX-MAX": "Agent (Claude 3.5 Sonnet, Max Cov.)",
    "edad/gen-claude-MIX-UNIF": "Agent (Claude 3.5 Sonnet, Unif.)",
    "edad/gen-gpt-MIX-MAX": "Agent (GPT-4o, Max Cov.)",
    "edad/gen-gpt-MIX-UNIF": "Agent (GPT-4o, Unif.)",
    "edad/gen-llama-MIX-MAX": "Agent (Llama 3.3 70B, Max Cov.)",
    "edad/gen-llama-MIX-UNIF": "Agent (Llama 3.3 70B, Unif.)",
    "edad/sdscm-gpt2": "SDSCM (GPT-2)",
    "edad/sdscm-llama-3-8b": "SDSCM (LLaMA-3 8B)",
    "edad/sdscm-olmo-1b-hf": "SDSCM (OLMO-1B HF)"
}


rename_map = {
    "pct_degradation_on_ref": "% Degradation on Ref."
}

method_grouping = {
    "Arbitrary": "Arbitrary",
    "Domain (Baseline)": "Baseline",
    "Univariate (Baseline)": "Baseline",
    "CSV (Claude 3.5 Sonnet)": "CSV",
    "CSV (GPT-4o)": "CSV",
    "CSV (Llama 3.3 70B)":  "CSV",
    "Agent (All, Max Cov.)": "Agent",
    "Agent (All, Unif.)": "Agent",
    "Agent (Llama 3.3 70B, Max Cov.)": "Agent",
    "Agent (Llama 3.3 70B, Unif.)": "Agent",
    "Agent (Claude 3.5 Sonnet, Max Cov.)": "Agent",
    "Agent (Claude 3.5 Sonnet, Unif.)": "Agent",
    "Agent (GPT-4o, Max Cov.)": "Agent",
    "Agent (GPT-4o, Unif.)": "Agent",
    "SDSCM (GPT-2)": "SDSCM",
    "SDSCM (LLaMA-3 8B)": "SDSCM",
    "SDSCM (OLMO-1B HF)":  "SDSCM",
    "Public": "Public",
    "Private": "Private"
}

grouping_colors = {
    # grey color
    "Baseline": "grey",
    'Arbitrary': 'blue',
    'Public': 'magenta',
    # green
    "CSV": "#228B22",
    # orange
    "Agent": "#FF8C00",
    # brown
    "SDSCM": "#8B4513",
    "Other": "black"
}

METRIC_GROUPS = {
    "marginals_metrics": [
        "marginals_3_max_abs_diff_error",
        "marginals_3_avg_abs_diff_error",
        "thresholded_marginals_3_max_abs_diff_error",
        "thresholded_marginals_3_avg_abs_diff_error",
    ],
    "classification_metrics": [
        "error_rate_diff",
        "auc_diff",
    ],
    "correlation_metrics": [
        "total_variation_distance",
        "pearson_corr_max_abs_diff",
        "pearson_corr_avg_abs_diff",
        "cramer_v_corr_max_abs_diff",
        "cramer_v_corr_avg_abs_diff",
    ],
}

metric_to_group = {}
for group_name, metrics_list in METRIC_GROUPS.items():
    for m in metrics_list:
        metric_to_group[m] = group_name

df = pd.DataFrame(all_results)

df_evaluation = pd.json_normalize(df["evaluation"])

def dict_to_sorted_json_str(d):
    return str(json.dumps(d, sort_keys=True)) # str(

df["hparams_str"] = df["hparams"].apply(dict_to_sorted_json_str)

df_combined = pd.concat(
    [df.drop(columns=["hparams", "evaluation"]),
     df_evaluation],
    axis=1
)

def get_reference_dataset_name(ds_name):
    if ds_name.startswith("acs/"):
        return reference_data_mapping["acs"]
    elif ds_name.startswith("edad/"):
        return reference_data_mapping["edad"]
    elif ds_name.startswith("we/"):
        return reference_data_mapping["we"]
    else:
        raise ValueError(f"unknown dataset name {ds_name}")

df_combined["reference_dataset_name"] = df_combined["dataset_name"].apply(get_reference_dataset_name)


def find_best_performance_rows(subdf, metric):
    direction = METRIC_DIRECTION[metric]
    if direction == "closer_to_zero_is_better":
        # for each hyperparam, compute mean(abs(metric))
        grouped_mean_abs = (
            subdf
            .groupby("hparams_str")[metric]
            .apply(lambda x: x.abs().mean())  # average over seeds
        )
        best_val = grouped_mean_abs.min()
        # if there's a tie, we just take the first
        best_hparams_strs = grouped_mean_abs[grouped_mean_abs == best_val].index
        best_hparams_str = best_hparams_strs[0]
        return best_val, best_hparams_str
    else:
        raise ValueError(f"everything is closer_to_zero_is_better now")
    
rows_for_report = []



# skipped tracker
skipped = {}
group_cols = ["synth_name", "epsilon", "reference_dataset_name"]
for (synth_name, epsilon, reference_dataset_name), group_df in df_combined.groupby(group_cols):
    # identify the reference dataset within this group
    ref_df = group_df[group_df["dataset_name"] == reference_dataset_name]
    if ref_df.empty:
        continue

    # find the 'true best performance' for each metric in that reference subset
    true_best_performance = {}
    for metric in METRIC_DIRECTION.keys():
        true_best_val, best_hparams = find_best_performance_rows(ref_df, metric)

        true_best_performance[metric] = (true_best_val, None, best_hparams)

    # for each dataset in the group, figure out which hyperparams you'd pick
    for ds_name, ds_group_df in group_df.groupby("dataset_name"):
        # then do that for each metric (or each metric block)
        for metric in METRIC_DIRECTION.keys():
            try:
                chosen_val_on_dataset, chosen_hparams_str = find_best_performance_rows(ds_group_df, metric)

                ref_match = ref_df[ref_df["hparams_str"] == chosen_hparams_str]
                if ref_match.empty:
                    # means reference never had that exact set of hyperparams
                    raise ValueError(f"reference dataset has no hyperparams {chosen_hparams_str} for metric {metric}")

                # how does it perform on the reference dataset?
                # before row_in_ref = ref_match.iloc[0]
                # instead, now, set row_in_ref
                perf_on_ref = ref_match[metric].abs().mean()

                # get the "true best" value for that metric
                (true_best_val, _, best_hparams) = true_best_performance[metric]

                # define percent_degradation = (candidate - best) / abs(best), if best != 0, else 0
                metric_group = metric_to_group[metric]
                if metric_group == "classification_metrics":
                    # for classification, do absolute difference
                    degradation_on_ref = abs(perf_on_ref - true_best_val)
                    # print('Classification')
                    # print(f'perf_on_ref: {perf_on_ref}')
                    # print(f'true_best_val: {true_best_val}')
                    # print(f"degradation_on_ref: {degradation_on_ref}")
                    # print()

                else:
                    if true_best_val == 0:
                        degradation_on_ref = 0
                    else:
                        degradation_on_ref = (perf_on_ref - true_best_val) / abs(true_best_val)

                # store in our report
                rows_for_report.append({
                    "synth_name": synth_name,
                    "epsilon": epsilon,
                    "dataset_name": ds_name,
                    "metric": metric,
                    "chosen_hparams_str": chosen_hparams_str,
                    "chosen_val_on_dataset": np.abs(perf_on_ref),
                    "perf_on_reference": perf_on_ref,
                    "true_best_on_reference": true_best_val,
                    "pct_degradation_on_ref": degradation_on_ref,
                    "best_hparams": best_hparams,
                    "reference_dataset_name": reference_dataset_name,
                })
            except Exception as e:
                print(f"skipping {synth_name}, {epsilon}, {ds_name}, {metric}: {e}")
                if (synth_name, epsilon, ds_name, metric) not in skipped:
                    skipped[(synth_name, epsilon, ds_name, metric)] = 0
                skipped[(synth_name, epsilon, ds_name, metric)] += 1

report_df = pd.DataFrame(rows_for_report)

# reverse mapping, metric -> group
metric_to_group = {}
for group, metrics in METRIC_GROUPS.items():
    for metric in metrics:
        metric_to_group[metric] = group

# metric_group to column to report_df
report_df['metric_group'] = report_df['metric'].map(metric_to_group)


def facet_plot_builder(df, method_name_map, plot_name="facet_plot"):
    df_for_plotting = df.copy()
    with plt.style.context(['science']):
        plt.rc('text', usetex=False)

        long_data = []
        for ds_name in df_for_plotting.index:
            row = df_for_plotting.loc[ds_name]

            long_data.append({
                "dataset_name": ds_name,
                "metric_group": "Classification",
                "mean": row["mean_class"],
                "std": row["std_class"],
            })
            long_data.append({
                "dataset_name": ds_name,
                "metric_group": "Correlation",
                "mean": row["mean_corr"],
                "std": row["std_corr"],
            })
            long_data.append({
                "dataset_name": ds_name,
                "metric_group": "Marginals",
                "mean": row["mean_marg"],
                "std": row["std_marg"],
            })

        plot_df = pd.DataFrame(long_data)

        # Map dataset_name -> label
        plot_df["dataset_name_label"] = (
            plot_df["dataset_name"]
            .map(method_name_map)
            .fillna(plot_df["dataset_name"])
        )

        # Map each label -> a grouping color, e.g. "Agent", "CSV", etc.
        plot_df["method_group"] = (
            plot_df["dataset_name_label"]
            .map(method_grouping)
            .fillna("Other")
        )

        # Sort so Classification is first, then Correlation, then Marginals
        # And to ensure bigger means come last in the barh (optional).
        plot_df = plot_df.sort_values(by=["metric_group", "mean"], ascending=[True, False])

        # Make the FacetGrid
        g = sns.FacetGrid(
            data=plot_df,
            col="metric_group",
            col_wrap=3,
            sharex=False,
            sharey=False,
            height=5
        )

        def barplot_with_errorbars(data, color=None, label=None, **kwargs):
            ax = plt.gca()
            for idx, row_ in data.iterrows():
                grp = row_["method_group"]
                c = grouping_colors.get(grp, grouping_colors['Other'])  # fallback color
                ax.barh(
                    y=row_["dataset_name_label"],
                    width=row_["mean"],
                    xerr=row_["std"],  # error bar
                    color=c,
                    capsize=0.1
                )

        g.map_dataframe(barplot_with_errorbars)

        for ax in g.axes.flat:
            for label_ in ax.get_yticklabels():
                label_.set_rotation(30)
            metric_group = ax.get_title().split(" = ")[1]

            ax.set_ylabel("")
            if metric_group == "Classification":
                ax.set_xlabel("Avg. Abs. Degradation")
            else:
                ax.set_xlabel("Avg. % Degradation")

            # y-axis label fontsize
            ax.yaxis.label.set_size(14)

            x_max = plot_df[plot_df["metric_group"] == metric_group]["mean"].max()
            ax.set_xlim(0, x_max)

            ax.set_title(metric_group)

        plt.tight_layout()
        plt.savefig(f"plots/{plot_name}.pdf", bbox_inches="tight")
        plt.close()  # close the figure so we can make a second one

        # group by metric_group + method_group, average 'mean' and 'std'
        # (we do a naive arithmetic mean of std, not the “standard error”.)
        agg_plot_df = (
            plot_df
            .groupby(["metric_group", "method_group"], as_index=False)
            .agg({"mean": "mean", "std": "mean"})
        )

        # want one FacetGrid again, but each col is metric_group,
        agg_plot_df = agg_plot_df.sort_values(by=["metric_group", "mean"], ascending=[True, False])

        g2 = sns.FacetGrid(
            data=agg_plot_df,
            col="metric_group",
            col_wrap=3,
            sharex=False,
            sharey=False,
            height=5
        )

        def barplot_with_errorbars_agg(data, color=None, label=None, **kwargs):
            ax = plt.gca()
            for idx, row_ in data.iterrows():
                grp = row_["method_group"]
                c = grouping_colors.get(grp, grouping_colors['Other'])  # fallback color
                ax.barh(
                    y=grp,            # method_group on y-axis
                    width=row_["mean"],
                    xerr=row_["std"],
                    color=c,
                    capsize=0.1
                )

        g2.map_dataframe(barplot_with_errorbars_agg)

        for ax in g2.axes.flat:
            for label_ in ax.get_yticklabels():
                label_.set_rotation(0)
            metric_group = ax.get_title().split(" = ")[1]

            ax.set_ylabel("Group")
            if metric_group == "Classification":
                ax.set_xlabel("Avg. Absolute Degradation (Aggregated)")
            else:
                ax.set_xlabel("Avg. % Degradation (Aggregated)")

            x_max = agg_plot_df[agg_plot_df["metric_group"] == metric_group]["mean"].max()
            ax.set_xlim(0, x_max)

            ax.set_title(metric_group)

        plt.tight_layout()
        plt.savefig(f"plots/{plot_name}_aggregated.pdf", bbox_inches="tight")
        plt.close()


def process_dataset(report_df, pattern, caption, label, synth_name='privbayes'):
    df_filtered = report_df[
        (report_df['synth_name'] == synth_name) 
        & (report_df['dataset_name'].str.contains(pattern))
    ]
    
    grouped_df_quick = (
        df_filtered
        .groupby(['dataset_name','metric_group'])
        .mean(numeric_only=True)
        .sort_values('pct_degradation_on_ref')
    )
    
    grouped_df_class = grouped_df_quick[
        grouped_df_quick.index.get_level_values('metric_group') == 'classification_metrics'
    ]
    print('Classification metrics')
    display(grouped_df_class)
    
    grouped_df_marg = grouped_df_quick[
        grouped_df_quick.index.get_level_values('metric_group') == 'marginals_metrics'
    ]
    print('Marginals metrics')
    display(grouped_df_marg)
    
    grouped_df_corr = grouped_df_quick[
        grouped_df_quick.index.get_level_values('metric_group') == 'correlation_metrics'
    ]
    print('Correlation metrics')
    display(grouped_df_corr)
    
    grouped_for_facet = (
        df_filtered
        .groupby(['dataset_name','metric_group'])['pct_degradation_on_ref']
        .agg(['mean','std'])
        .sort_values('mean')
    )
    
    # NOTE: each aggregated row was computed from at least 5 seeds, which is washed out in some of the
    # aggregation - we correct as follows
    # => standard error = std / sqrt(5)
    grouped_for_facet['std'] = grouped_for_facet['std'] / np.sqrt(5)

    gdcl = grouped_for_facet[grouped_for_facet.index.get_level_values('metric_group') == 'classification_metrics']
    gdco = grouped_for_facet[grouped_for_facet.index.get_level_values('metric_group') == 'correlation_metrics']
    gdcm = grouped_for_facet[grouped_for_facet.index.get_level_values('metric_group') == 'marginals_metrics']
    
    facet_df = gdcl.merge(gdco, on='dataset_name', suffixes=('_class', '_corr'))
    facet_df = facet_df.merge(gdcm, on='dataset_name')
    facet_df = facet_df.rename(columns={"mean": "mean_marg", "std": "std_marg"})
    
    for col in ['mean_class','std_class','mean_corr','std_corr','mean_marg','std_marg']:
        facet_df[col] = facet_df[col] * 100
    
    drop_cols = ['epsilon', 'chosen_val_on_dataset', 'perf_on_reference', 'true_best_on_reference']
    grouped_df_class = grouped_df_class.drop(columns=drop_cols, errors='ignore')
    grouped_df_corr = grouped_df_corr.drop(columns=drop_cols, errors='ignore')
    grouped_df_marg = grouped_df_marg.drop(columns=drop_cols, errors='ignore')
    
    latex_df = grouped_df_class.merge(grouped_df_corr, on='dataset_name', suffixes=('_class', '_corr'))
    latex_df = latex_df.merge(grouped_df_marg, on='dataset_name')
    latex_df = latex_df.rename(columns={"pct_degradation_on_ref": "pct_degradation_on_ref_marg"})
    
    latex_df = latex_df[~latex_df.index.str.contains('national')]
    latex_df = latex_df[~latex_df.index.str.contains('2023')]
    
    latex_df = latex_df.rename(columns={
        'pct_degradation_on_ref_class': r'\% Degradation (Class.)',
        'pct_degradation_on_ref_corr':  r'\% Degradation (Corr.)',
        'pct_degradation_on_ref_marg':  r'\% Degradation (Marg.)'
    })
    
    latex_table = df_to_latex_medals(latex_df, caption=caption, label=label, columns_to_exclude=[])
    print(latex_table)
    # write latex_table to file in tables
    with open(f"tables/{label.split(':')[1]}_{synth_name}.tex", "w") as f:
        f.write(latex_table)

    return facet_df, latex_df


def identify_pareto_frontier(df, metric_columns):
    costs = df[metric_columns].values
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Any point dominated by point i is not efficient
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

def pareto_frontier_analysis_for_df(df, synth_name='privbayes', dataset_name='acs'):
    from pandas.plotting import parallel_coordinates

    agg_df = df.groupby(['dataset_name', 'metric_group'])['pct_degradation_on_ref'].mean().reset_index()

    pivot_df = agg_df.pivot(index='dataset_name', columns='metric_group', values='pct_degradation_on_ref').reset_index()

    pivot_df.columns.name = None

    pivot_df.rename(
        columns={
            'classification_metrics': 'classification',
            'marginals_metrics': 'marginals',
            'correlation_metrics': 'correlation'
        }, 
        inplace=True
    )

    excludes = ['2018', '2023', 'national', 'massachusetts']
    for ex in excludes:
        pivot_df = pivot_df[~pivot_df['dataset_name'].str.contains(ex)]

    metric_cols = ['classification', 'marginals', 'correlation']
    pivot_df['pareto'] = identify_pareto_frontier(pivot_df, metric_cols)

    pareto_df = pivot_df[pivot_df['pareto']]
    print("\nThese are pareto-efficient methods!")
    print(pareto_df, "\n")

    mapping_prefixes = {
        **we_method_name_map,
        **acs_method_name_map,
        **edad_method_name_map
    }
    latex_table = create_latex_table_with_medals(pareto_df.copy().drop(columns=['pareto']), 
                                                 caption=f"Pareto Efficient Methods for {synth_name} on {dataset_name}",
                                                 label=f"tab:pareto_efficient_methods_{synth_name}_{dataset_name}",
                                                 largest_is_better=False, mapping_prefixes=mapping_prefixes)
    print(latex_table)
    with open(f"tables/pareto_efficient_methods_{synth_name}_{dataset_name}.tex", "w") as f:
        f.write(latex_table)

    with plt.style.context(['science']):
        plt.rc('text', usetex=False)
        pivot_df['pareto_front'] = pivot_df['pareto'].apply(lambda x: 'pareto front' if x else 'non-pareto')
        plt.figure(figsize=(8, 4))
        pivot_df = pivot_df.sort_values('pareto_front', ascending=False)

        parallel_coordinates(
            pivot_df,
            class_column='pareto_front',
            cols=['classification', 'marginals', 'correlation'],
            color=('blue', 'grey'),
            linewidth=2
        )
        plt.title('Parallel Coordinates Plot (Pareto Frontier)', fontsize=15)
        plt.xlabel('Metric Groups', fontsize=12)
        plt.ylabel('% Degradation', fontsize=12)
        plt.yscale('log')
        plt.legend(title='Pareto Class')
        plt.xticks(ticks=[0, 1, 2], labels=['Classification', 'Marginals', 'Correlation'])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/pareto_frontier_{synth_name}_{dataset_name}.pdf", bbox_inches="tight")


# ACS
merged_df_acs, grouped_df_acs = process_dataset(
    report_df, 
    pattern='acs', 
    caption="ACS Metrics", 
    label="tab:all_metrics_acs_privbayes",
    synth_name='privbayes'
)
facet_plot_builder(merged_df_acs, acs_method_name_map, plot_name="facet_plot_acs_privbayes")

# EDAD
merged_df_edad, grouped_df_edad = process_dataset(
    report_df,
    pattern='edad',
    caption="EDAD Metrics",
    label="tab:all_metrics_edad_privbayes",
    synth_name='privbayes'
)
facet_plot_builder(merged_df_edad, edad_method_name_map, plot_name="facet_plot_edad_privbayes")

# WE
merged_df_we, grouped_df_we = process_dataset(
    report_df,
    pattern='we',
    caption="WE Metrics",
    label="tab:all_metrics_we_privbayes",
    synth_name='privbayes'
)
facet_plot_builder(merged_df_we, we_method_name_map, plot_name="facet_plot_we_privbayes")

# ACS
merged_df_acs, grouped_df_acs = process_dataset(
    report_df, 
    pattern='acs', 
    caption="ACS Metrics", 
    label="tab:all_metrics_acs_jax",
    synth_name='aim_jax'
)
facet_plot_builder(merged_df_acs, acs_method_name_map, plot_name="facet_plot_acs_aim_jax")

# EDAD
merged_df_edad, grouped_df_edad = process_dataset(
    report_df,
    pattern='edad',
    caption="EDAD Metrics",
    label="tab:all_metrics_edad_jax",
    synth_name='aim_jax'
)
facet_plot_builder(merged_df_edad, edad_method_name_map, plot_name="facet_plot_edad_aim_jax")

# WE
merged_df_we, grouped_df_we = process_dataset(
    report_df,
    pattern='we',
    caption="WE Metrics",
    label="tab:all_metrics_we_jax",
    synth_name='aim_jax'
)
facet_plot_builder(merged_df_we, we_method_name_map, plot_name="facet_plot_we_aim_jax")


# ACS
merged_df_acs, grouped_df_acs = process_dataset(
    report_df, 
    pattern='acs', 
    caption="ACS Metrics", 
    label="tab:all_metrics_acs_gem",
    synth_name='gem'
)
facet_plot_builder(merged_df_acs, acs_method_name_map, plot_name="facet_plot_acs_gem")

# EDAD
merged_df_edad, grouped_df_edad = process_dataset(
    report_df,
    pattern='edad',
    caption="EDAD Metrics",
    label="tab:all_metrics_edad_gem",
    synth_name='gem'
)
facet_plot_builder(merged_df_edad, edad_method_name_map, plot_name="facet_plot_edad_gem")

# WE
merged_df_we, grouped_df_we = process_dataset(
    report_df,
    pattern='we',
    caption="WE Metrics",
    label="tab:all_metrics_we_gem",
    synth_name='gem'
)
facet_plot_builder(merged_df_we, we_method_name_map, plot_name="facet_plot_we_gem")

pareto_frontier_analysis_for_df(report_df[(report_df['synth_name'] == 'gem') & (report_df['dataset_name'].str.contains('acs'))], synth_name='gem', dataset_name='acs')
pareto_frontier_analysis_for_df(report_df[(report_df['synth_name'] == 'gem') & (report_df['dataset_name'].str.contains('edad'))], synth_name='gem', dataset_name='edad')
pareto_frontier_analysis_for_df(report_df[(report_df['synth_name'] == 'gem') & (report_df['dataset_name'].str.contains('we'))], synth_name='gem', dataset_name='we')

pareto_frontier_analysis_for_df(report_df[(report_df['synth_name'] == 'privbayes') & (report_df['dataset_name'].str.contains('acs'))], synth_name='privbayes', dataset_name='acs')
pareto_frontier_analysis_for_df(report_df[(report_df['synth_name'] == 'privbayes') & (report_df['dataset_name'].str.contains('edad'))], synth_name='privbayes', dataset_name='edad')
pareto_frontier_analysis_for_df(report_df[(report_df['synth_name'] == 'privbayes') & (report_df['dataset_name'].str.contains('we'))], synth_name='privbayes', dataset_name='we')

pareto_frontier_analysis_for_df(report_df[(report_df['synth_name'] == 'aim_jax') & (report_df['dataset_name'].str.contains('acs'))], synth_name='aim_jax', dataset_name='acs')
pareto_frontier_analysis_for_df(report_df[(report_df['synth_name'] == 'aim_jax') & (report_df['dataset_name'].str.contains('edad'))], synth_name='aim_jax', dataset_name='edad')
pareto_frontier_analysis_for_df(report_df[(report_df['synth_name'] == 'aim_jax') & (report_df['dataset_name'].str.contains('we'))], synth_name='aim_jax', dataset_name='we')
