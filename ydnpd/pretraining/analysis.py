from typing import List, Optional
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'no-latex'])


PUBLIC_DATASETS = ["acs/massachusetts_upsampled", "edad/2020", "we/2018"]
CATEGORY_ORDER = ["Without pretraining", "Public", "Baseline", "Arbitrary", "CSV", "Agent"]  #"SD-SCM"]

LIMS = {
    "auc": (0.5, 0.86),
    "adv": (-0.07, 0.2)
}

    # Define color mapping
color_map = {
    'Without pretraining': 'red',
    'Baseline': 'grey',
    'Arbitrary': 'blue',
    'Public': 'magenta',
    'Arbitrary': 'blue',                 # Keep blue
    'CSV': '#228B22',                    # Green (X)
    "Agent": '#FF8C00',                  # Orange (Y)
    # 'SD-SCM': '#8B4513'                  # Brown (Z)
}


# Define category mapping function
def get_category(pointer, data_prefix):
    if f'{data_prefix}/no-pre' in pointer:
        return 'Without pretraining'
    elif pointer in PUBLIC_DATASETS:
        return "Public"
    elif f'{data_prefix}/baseline' in pointer:
        return 'Baseline'
    elif f'{data_prefix}/arbitrary' in pointer:
        return 'Arbitrary'
    elif f'{data_prefix}/csv' in pointer:
        return 'CSV'
    elif f'{data_prefix}/gen' in pointer:
        return 'Agent'
    # elif f'{data_prefix}/sdscm' in pointer:
        # return 'SD-SCM'


def extract_method_label(r):
    model_format = {"gpt": "GPT-4o",
    "gpt2": "GPT-2",
    "claude": "Claude 3.5 Sonnet",
    "llama": "Llama 3.3 70B",
    "llama-3-8b": "Llama 3.1 8B",
    "olmo-1b-hf": "OLMo 1B"
    }
    dataset_name = r["public_data_pointer"].split("/")[1]

    if dataset_name.startswith("csv"):
        name = model_format[dataset_name.split("-")[1]]
    
    # elif dataset_name.startswith("sdscm"):
    #     name = model_format[dataset_name.split("-", 1)[1]]

    elif dataset_name.startswith("gen"):
        parts = dataset_name[4:].split("MIX")
        model = parts[0][:-1].title()
        name =  (
            (model_format[parts[0][:-1]]if parts[0] else "")
            + ("," if parts[0] else "")
            + f" MIX{parts[1]}"
            ).strip()

        name = (name
            .replace("MIX-MAX", "Max Cov.")
            .replace("MIX-UNIF", "Unif.")
        )

    elif dataset_name.startswith("baseline"):
        name = dataset_name.split("_")[1].title()

    else:
        name = ""

    return r["category"] + (f" ({name})" if name else "")


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate the input DataFrame has required columns and format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate
    
    Raises
    ------
    ValueError
        If DataFrame doesn't meet requirements
    """
    if 'category' not in df.columns:
        raise ValueError("DataFrame must contain 'category' column")
    
    # Check for epsilon columns
    eps_cols = [col for col in df.columns if col.startswith('eps=')]
    if not eps_cols:
        raise ValueError("DataFrame must contain epsilon columns (format: 'eps=X')")
    
    # Verify epsilon columns contain numeric data
    for col in eps_cols:
        if not pd.to_numeric(df[col], errors='coerce').notna().all():
            raise ValueError(f"Column {col} must contain only numeric values")

def get_epsilon_columns(df: pd.DataFrame) -> List[str]:
    """Extract and sort epsilon columns from DataFrame."""
    eps_cols = [col for col in df.columns if col.startswith('eps=')]
    
    # Extract numeric values and sort
    def get_eps_value(col: str) -> float:
        return float(col.split('=')[1])
    
    return sorted(eps_cols, key=get_eps_value)

def get_color_coding(val: float, col_values: np.ndarray) -> str:
    """
    Get LaTeX color coding based on value's rank.
    
    Parameters
    ----------
    val : float
        Value to check
    col_values : np.ndarray
        All values in the column
    
    Returns
    -------
    str
        LaTeX color command or empty string
    """
    sorted_vals = sorted(set(col_values), reverse=True)
    rank = sorted_vals.index(val)
    
    color_map = {
        0: 'gold',
        1: 'silver',
        2: 'bronze'
    }
    
    return f'\\cellcolor{{{color_map[rank]}!30}}' if rank in color_map else ''

# def generate_latex_table(
#     df: pd.DataFrame,
#     caption: str = "Results by method and epsilon value",
#     label: str = "tab:epsilon_comparison"
# ) -> str:
#     """
#     Generate a LaTeX table from a pandas DataFrame with color-coding for top 3 values.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Must contain:
#             - 'category' column
#             - 'eps=X' columns with numeric values
#         Optional:
#             - 'Method' column for detailed tables
#     caption : str
#         Table caption
#     label : str
#         Table label for referencing
    
#     Returns
#     -------
#     str
#         LaTeX table code
#     """
#     # Validate input
#     validate_dataframe(df)
    
#     # Determine table type and get epsilon columns
#     is_detailed = 'Method' in df.columns
#     eps_cols = get_epsilon_columns(df)
    
#     # Start building table
#     latex = [
#         '\\begin{table}[h!]',
#         '    \\centering',
#         f'    \\caption{{{caption}}}',
#         f'    \\label{{{label}}}',
#         '    \\begin{tabular}{l' + 'c' * len(eps_cols) + '}',
#         '    \\toprule'
#     ]
    
#     # Header row
#     header = ['Method Category' if not is_detailed else 'Method']
#     header.extend([f'$\\varepsilon={col.split("=")[1]}$' for col in eps_cols])
#     latex.append('    ' + ' & '.join(header) + ' \\\\')
#     latex.append('    \\midrule')
    
#     # Process data rows
#     prev_category = None
#     for idx, row in df.iterrows():
#         # Add category separator if needed
#         if ((is_detailed and prev_category is not None and row['category'] != prev_category)
#         or (not is_detailed and prev_category in {"Without pretraining", "Public", "Baseline", "Arbitrary"})):
#             latex.append('    \\arrayrulecolor{black!50!}\\midrule')
        
#         # Build row content
#         row_latex = [row['Method'] if is_detailed else row['category']]
#         for col in eps_cols:
#             val = row[col]
#             color = get_color_coding(val, df[col].values)
#             row_latex.append(f'{color}{val:.3f}')
#         latex.append('    ' + ' & '.join(row_latex) + ' \\\\')
        
#         # if is_detailed:
#         prev_category = row['category']
    
#     # Close table
#     latex.extend([
#         '    \\bottomrule',
#         '    \\end{tabular}',
#         '\\end{table}'
#     ])
    
#     return '\n'.join(latex)

def generate_latex_table(
    df: pd.DataFrame,
    advantage_df: pd.DataFrame,
    caption: str = "Results by method and epsilon value",
    label: str = "tab:epsilon_comparison"
) -> str:
    """
    Generate a LaTeX table from a pandas DataFrame with color-coding for top 3 values.
    Each cell contains advantage value and the AUC value in parentheses (smaller font).
    
    Parameters
    ----------
    df : pd.DataFrame
        Original AUC values. Must contain:
            - 'category' column
            - 'eps=X' columns with numeric values
        Optional:
            - 'Method' column for detailed tables
    advantage_df : pd.DataFrame
        Advantage values (matching structure to df)
    caption : str
        Table caption
    label : str
        Table label for referencing
    
    Returns
    -------
    str
        LaTeX table code
    """
    # Validate input
    validate_dataframe(df)
    
    # Determine table type and get epsilon columns
    is_detailed = 'Method' in df.columns
    eps_cols = get_epsilon_columns(df)
    
    # Create lookup for advantage values
    advantage_lookup = {}
    for idx, row in advantage_df.iterrows():
        category = row['category']
        method = row.get('Method', None)  # Get Method if it exists
        
        key = (category, method) if is_detailed else category
        advantage_lookup[key] = {col: row[col] for col in eps_cols}
    
    # Start building table
    latex = [
        '\\begin{table}[h!]',
        '    \\centering',
        f'    \\caption{{{caption}}}',
        f'    \\label{{{label}}}',
        '    \\begin{tabular}{l' + 'c' * len(eps_cols) + '}',
        '    \\toprule'
    ]
    
    # Header row
    header = ['Method Category' if not is_detailed else 'Method']
    header.extend([f'$\\varepsilon={col.split("=")[1]}$' for col in eps_cols])
    latex.append('    ' + ' & '.join(header) + ' \\\\')
    latex.append('    \\midrule')
    
    # Process data rows
    prev_category = None
    for idx, row in df.iterrows():
        # Add category separator if needed
        if ((is_detailed and prev_category is not None and row['category'] != prev_category)
        or (not is_detailed and prev_category in {"Without pretraining", "Public", "Baseline", "Arbitrary"})):
            latex.append('    \\arrayrulecolor{black!50!}\\midrule')
        
        # Build row content
        row_latex = [row['Method'] if is_detailed else row['category']]
        
        for col in eps_cols:
            auc_val = row[col]
            
            # Get advantage value
            key = (row['category'], row.get('Method', None)) if is_detailed else row['category']
            advantage_val = 0.0  # Default for "Without pretraining"
            
            if key in advantage_lookup:
                advantage_val = advantage_lookup[key].get(col, 0.0)
            
            # Format values - round to 2 decimal places and handle special cases
            # Drop leading zero for AUC value
            auc_formatted = f"{auc_val:.2f}".replace("0.", ".")
            
            # Handle negative zero case in advantage values and use .XX format
            if abs(advantage_val) < 0.005:  # This will catch -0.00 cases
                advantage_formatted = ".00"
            else:
                # Remove leading zero for decimal values
                advantage_formatted = f"{advantage_val:.2f}".replace("0.", ".")
            
            # Get color coding based on advantage value (for non-baseline)
            color = ""
            if row['category'] != "Without pretraining":
                # Get all advantage values for this column except "Without pretraining"
                all_adv_values = [v.get(col, 0.0) for k, v in advantage_lookup.items() 
                                 if (isinstance(k, tuple) and k[0] != "Without pretraining") or 
                                    (not isinstance(k, tuple) and k != "Without pretraining")]
                color = get_color_coding(advantage_val, all_adv_values)
            
            # Construct cell with advantage value and AUC in smaller parentheses
            cell = f"{color}{advantage_formatted} {{\small ({auc_formatted})}}"
            row_latex.append(cell)
        
        latex.append('    ' + ' & '.join(row_latex) + ' \\\\')
        
        prev_category = row['category']
    
    # Close table
    latex.extend([
        '    \\bottomrule',
        '    \\end{tabular}',
        '\\end{table}'
    ])
    
    return '\n'.join(latex)


METHOD_ORDER = ['Without pretraining',
 'Public',
 'Baseline (Domain)',
 'Baseline (Univariate)',
 'Arbitrary',
 'CSV (Claude 3.5 Sonnet)',
 'CSV (GPT-4o)',
 'CSV (Llama 3.3 70B)',
 'Agent (Claude 3.5 Sonnet, Unif.)',
 'Agent (Claude 3.5 Sonnet, Max Cov.)',
 'Agent (GPT-4o, Unif.)',
 'Agent (GPT-4o, Max Cov.)',
 'Agent (Llama 3.3 70B, Unif.)',
 'Agent (Llama 3.3 70B, Max Cov.)',
 'Agent (Unif.)',
 'Agent (Max Cov.)',
#  'SD-SCM (GPT-2)',
#  'SD-SCM (Llama 3.1 8B)',
#  'SD-SCM (OLMo 1B)'
]

# def build_table(plot_df, viz_unit, x_axis):

#     paper_df = (plot_df
#     .groupby([x_axis, 'category'] + (['public_data_pointer'] if viz_unit == "dataset" else []))[
#         "dp/private.test/auc"]
#         .mean()
#         .round(3)
#         .unstack(level=0)
#         .reset_index()
#         )

#     if viz_unit == "dataset":
#         df = (paper_df
#         .assign(Method=pd.Categorical(paper_df.apply(extract_method_label, axis=1), categories=METHOD_ORDER, ordered=True))
#         .drop(columns=["public_data_pointer"])
#         .sort_values("Method")
#         .reset_index(drop=True)
#         )

#         df = df[["category", "Method"] + [col for col in df.columns if col not in {"Method", "category"}]]
#         df.columns = ["category", "Method"] + [fr"eps={col}" for col in df.columns[2:]]

#     else:
#         df = (paper_df
#         .assign(category=pd.Categorical(paper_df["category"], categories=CATEGORY_ORDER, ordered=True))
#         .sort_values("category")
#         .reset_index(drop=True)
#         )

#         df = df[["category"] + [col for col in df.columns if col not in {"category"}]]
#         df.columns = ["category"] + [fr"eps={col}" for col in df.columns[1:]]


#     return df


def build_table(plot_df, viz_unit, x_axis):
    # First, compute the original AUC table
    paper_df = (plot_df
        .groupby([x_axis, 'category'] + (['public_data_pointer'] if viz_unit == "dataset" else []))[
            "dp/private.test/auc"]
            .mean()
            .round(3)
            .unstack(level=0)
            .reset_index()
    )

    # Build the original table
    if viz_unit == "dataset":
        df = (paper_df
            .assign(Method=pd.Categorical(paper_df.apply(extract_method_label, axis=1), categories=METHOD_ORDER, ordered=True))
            .drop(columns=["public_data_pointer"])
            .sort_values("Method")
            .reset_index(drop=True)
        )

        df = df[["category", "Method"] + [col for col in df.columns if col not in {"Method", "category"}]]
        df.columns = ["category", "Method"] + [fr"eps={col}" for col in df.columns[2:]]

    else:
        df = (paper_df
            .assign(category=pd.Categorical(paper_df["category"], categories=CATEGORY_ORDER, ordered=True))
            .sort_values("category")
            .reset_index(drop=True)
        )

        df = df[["category"] + [col for col in df.columns if col not in {"category"}]]
        df.columns = ["category"] + [fr"eps={col}" for col in df.columns[1:]]

    # Now, compute the advantage table using the same approach
    
    # First, get the raw data grouped the same way as the original table
    adv_plot_df = plot_df.copy()
    
    # Calculate the "Without pretraining" reference values for each epsilon
    baseline_data = {}
    for eps_val in adv_plot_df[x_axis].unique():
        baseline_mask = (adv_plot_df[x_axis] == eps_val) & (adv_plot_df['category'] == 'Without pretraining')
        if baseline_mask.any():
            baseline_data[eps_val] = adv_plot_df.loc[baseline_mask, 'dp/private.test/auc'].mean()
    
    # Apply the advantage calculation: subtract baseline from each corresponding epsilon value
    for eps_val, baseline_value in baseline_data.items():
        # For non-baseline categories, subtract the baseline value
        mask = (adv_plot_df[x_axis] == eps_val) & (adv_plot_df['category'] != 'Without pretraining')
        adv_plot_df.loc[mask, 'dp/private.test/auc'] -= baseline_value
        
        # For the "Without pretraining" category, set values to 0
        mask_baseline = (adv_plot_df[x_axis] == eps_val) & (adv_plot_df['category'] == 'Without pretraining')
        adv_plot_df.loc[mask_baseline, 'dp/private.test/auc'] = 0.0
    
    # Keep "Without pretraining" rows, but they will be zeros
    
    # Now follow the same approach as the original table
    adv_paper_df = (adv_plot_df
        .groupby([x_axis, 'category'] + (['public_data_pointer'] if viz_unit == "dataset" else []))[
            "dp/private.test/auc"]
            .mean()
            .round(3)
            .unstack(level=0)
            .reset_index()
    )
    
    # Build the advantage table with the same structure as the original
    if viz_unit == "dataset":
        adv_df = (adv_paper_df
            .assign(Method=pd.Categorical(adv_paper_df.apply(extract_method_label, axis=1), categories=METHOD_ORDER, ordered=True))
            .drop(columns=["public_data_pointer"])
            .sort_values("Method")
            .reset_index(drop=True)
        )

        adv_df = adv_df[["category", "Method"] + [col for col in adv_df.columns if col not in {"Method", "category"}]]
        adv_df.columns = ["category", "Method"] + [fr"eps={col}" for col in adv_df.columns[2:]]

    else:
        adv_df = (adv_paper_df
            .assign(category=pd.Categorical(adv_paper_df["category"], categories=CATEGORY_ORDER, ordered=True))
            .sort_values("category")
            .reset_index(drop=True)
        )

        adv_df = adv_df[["category"] + [col for col in adv_df.columns if col not in {"category"}]]
        adv_df.columns = ["category"] + [fr"eps={col}" for col in adv_df.columns[1:]]
    
    return df, adv_df

def plot_metrics(data_prefix, runs_df, viz_unit, is_adv=False, no_dp_results=None,
                caption=None, x_axis="epsilon",
                 with_legend=True, logscale=True, figsize=(10, 6)):
    """
    Create a line plot showing AUC vs Epsilon for different dataset categories,
    with a horizontal line showing the non-DP baseline performance.
    """


    factor = 1.25 if is_adv else 1
    plt.rcParams.update({
    'font.size': 18 * factor,  # Increase from 14
    'axes.labelsize': 20 * factor,  # Increase from 16
    'axes.titlesize': 20 * factor,  # Increase from 16
    'xtick.labelsize': 18 * factor,  # Increase from 14
    'ytick.labelsize': 18 * factor,  # Increase from 14
    'legend.fontsize': 18 * factor,  # Increase from 14
    })


    if viz_unit not in {"dataset", "category"}:
        raise ValueError("viz_unit")

    if is_adv and viz_unit == "dataset":
        raise ValueError("Cannot plot advantage results by dataset")

    
    LINEWIDTH_LARGE, LINEWIDTH_SMALL = 4.0, 2.0
    linewidth_not_core = LINEWIDTH_SMALL if viz_unit == "dataset" else LINEWIDTH_LARGE
    alpha_not_core = 0.7 if viz_unit == "dataset" else 1

    linewidth_map = {
        'Without pretraining': LINEWIDTH_LARGE,
        'Baseline': LINEWIDTH_LARGE,
        'Public': LINEWIDTH_LARGE,
        'Arbitrary': linewidth_not_core,
        'CSV': linewidth_not_core,
        "Agent": linewidth_not_core,
        # 'SD-SCM': linewidth_not_core,
    }

    alpha_map = {
        'Without pretraining': 1,
        'Baseline': 1,
        'Public': 1,
        'Arbitrary': 1,
        'CSV': alpha_not_core,
        "Agent": alpha_not_core,
        # 'SD-SCM': alpha_not_core
    }
    def plotter(plot_data):

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.75, 0.8]) 

        # Plot each category separately
        legend_handles = []
        legend_labels = []
        
        updated_category_order = CATEGORY_ORDER[:]

        if viz_unit == "dataset":
            plot_data = (plot_data
                        .groupby(['category', 'public_data_pointer', x_axis])
                        ['dp/private.test/auc']
                        .agg(['mean', 'std', 'size'])
                        .rename({"mean": "dp/private.test/auc"}, axis=1)
                        .reset_index())

            # Calculate standard error using the actual sample size for each group
            plot_data['se'] = plot_data['std'] / np.sqrt(plot_data['size'])
            estimator, errorbar, units = None, None, "public_data_pointer"
            
        elif viz_unit == "category" and not is_adv:
            estimator, errorbar, units = "mean", ("ci", 95), None
        
        else:
            estimator, errorbar, units = None, None, None
            plot_data = plot_data.groupby(["category", x_axis])["dp/private.test/auc"].mean().reset_index()
            no_pre = plot_data.loc[plot_data["category"] == "Without pretraining"]
            plot_data = plot_data.loc[plot_data["category"] != "Without pretraining"]
            for x_value in plot_data[x_axis].unique():
                no_pre_value = no_pre.loc[no_pre[x_axis] == x_value, "dp/private.test/auc"].values[0]
                plot_data.loc[plot_data[x_axis] == x_value, "dp/private.test/auc"] -= no_pre_value
            updated_category_order.remove("Without pretraining")     

        for category in updated_category_order:
            category_data = plot_data[plot_data['category'] == category]
            
            _ = sns.lineplot(
                data=category_data,
                x=x_axis,
                y='dp/private.test/auc',
                units=units,
                estimator=estimator,
                errorbar=errorbar,
                err_style="bars", #errorbar=("se", 2),
                color=color_map[category],
                alpha=alpha_map[category],
                linewidth=linewidth_map[category],
                marker='o',
                markersize=10,
                label='_nolegend_',  # Suppress automatic legend entries
                ax=ax
            )

            legend_handles.append(plt.Line2D([0], [0], color=color_map[category], 
                                        alpha=alpha_map[category],
                                        linewidth=linewidth_map[category],
                                        marker='o', markersize=6,
                                        markerfacecolor=color_map[category]))
            legend_labels.append(category)

        # Customize the plot
        if logscale:
            ax.set_xscale('log')
        ax.set_xlabel(x_axis.title())
        ax.set_ylabel('AUC' + (' Advantage' if is_adv else ''))
        # ax.set_title(f'AUC vs Epsilon by Dataset ({data_prefix.upper()} Data)')

        if is_adv:
            ax.set_ylim(LIMS['adv'])
        else:
            ax.set_ylim(LIMS['auc'])

        # Set x-axis ticks
        unique_x_axis_values = sorted(plot_data[x_axis].unique())
        ax.set_xticks(unique_x_axis_values)

        # Format tick labels to drop leading zeros for float values
        formatted_xticks = []
        for val in unique_x_axis_values:
            if isinstance(val, float):
                if 0 < val < 1:
                    # Drop the leading zero for decimal values less than 1
                    # Use the original precision (don't round)
                    val_str = str(val).replace("0.", ".")
                    formatted_xticks.append(val_str)
                elif val.is_integer():
                    # Convert float values like 1.0, 2.0 to integers
                    formatted_xticks.append(str(int(val)))
                else:
                    # Keep other float values as they are
                    formatted_xticks.append(str(val))
            else:
                # Keep non-float values as they are
                formatted_xticks.append(str(val))

        ax.set_xticklabels(formatted_xticks)
        # ax.set_xticklabels(unique_x_axis_values)
        
        # Add horizontal line for no-DP baseline if needed
        if no_dp_results is not None and data_prefix in no_dp_results:
            baseline_auc = np.mean([result['test/auc'] for result in no_dp_results[data_prefix]])
            ax.axhline(y=baseline_auc, color='black', linestyle='--')
            # Add baseline to legend
            legend_handles.append(plt.Line2D([0], [0], color='black', linestyle='--'))
            legend_labels.append('No DP baseline')

        # Set legend with custom handles
        # ax.legend(legend_handles, legend_labels)

        if is_adv:
            hline_style = {'linestyle': '--', 'color': 'red', 'linewidth': LINEWIDTH_LARGE}
            ax.axhline(y=0, **hline_style)
            legend_handles.append(plt.Line2D([0], [0], **hline_style))
            legend_labels.append("Without pretraining")


        if with_legend:
            ax.legend(legend_handles, legend_labels,
                bbox_to_anchor=(1.05, 0.5),
                loc='center left',
                borderaxespad=0,
                frameon=False)

        plt.tight_layout()  # Adjusts layout

        return fig

        # Filter and prepare data
    plot_data = (runs_df
        .loc[runs_df['private_data_pointer'].str.startswith(data_prefix)]
    )
    plot_data.loc[:, 'category'] = plot_data.loc[:, 'public_data_pointer'].apply(lambda x: get_category(x, data_prefix))    

    fig = plotter(plot_data)

    auc_paper_df, advantage_paper_df = build_table(plot_data, viz_unit, x_axis)
    print(auc_paper_df)
    print(advantage_paper_df)
    tab = generate_latex_table(auc_paper_df, advantage_paper_df, caption=caption) if x_axis == "epsilon" else None

    return fig, tab, plot_data

def create_caption(dataset_name, viz_unit, is_adv=False, is_best=False):
    return f"Mean AUC {'Advantage ' if is_adv else ''}of the DP model on {dataset_name.upper()} data after pretraining, grouped by generation method{' category' if viz_unit == 'category' else ''}. The mean is calculated across the {'DP finetuning ' if is_best else ''}hyperparameter space{' when best pretraining hyperparameter configuration is choosen for the pretraining step' if is_best else ''}, with 10 runs per hyperparameter configuration."