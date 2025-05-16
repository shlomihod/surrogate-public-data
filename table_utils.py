import pandas as pd
import numpy as np

def df_to_latex_medals(
    df,
    caption = "default caption",
    label = "tab:default",
    float_format = ".3f",
    largest_is_better = False,
    index_name = "Method",
    columns_to_exclude = []
) -> str:
    # columns to exclude
    df = df.drop(columns=columns_to_exclude, errors='ignore', axis=1)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    ranks = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        ranks[col] = df[col].rank(method='dense', ascending=not largest_is_better)

    lines = []

    # lines.append(r"\begin{table}[t!]")
    # lines.append(r"    \centering")
    # lines.append(f"    \\caption{{{caption}}}")
    # lines.append(f"    \\label{{{label}}}")
    # NOTE: removed resizebox
    # lines.append(r"    \resizebox{\linewidth}{!}{%")
    lines.append(r"    \begin{tabular}{l" + "c" * len(df.columns) + "}")
    lines.append(r"    \toprule")

    header_cols = [index_name] + [str(col) for col in df.columns]
    lines.append("    " + " & ".join(header_cols) + r" \\")
    lines.append(r"    \midrule")

    for idx in df.index:
        row_vals = []
        row_vals.append(str(idx))

        for col in df.columns:
            val = df.loc[idx, col]

            # if col in numeric_cols and pd.notnull(val):
            if pd.notnull(df[col]).all():
                val_str = f"{val:{float_format}}"

                rank_val = ranks.loc[idx, col]
                if rank_val == 1:
                    val_str = r"\cellcolor{gold!30}" + val_str
                elif rank_val == 2:
                    val_str = r"\cellcolor{silver!30}" + val_str
                elif rank_val == 3:
                    val_str = r"\cellcolor{bronze!30}" + val_str

                row_vals.append(val_str)
            else:
                row_vals.append(str(val))

        line = " & ".join(row_vals) + r" \\"
        lines.append("    " + line)

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}") # NOTE: removed resizebox
    # lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    return latex_str

def create_latex_table_with_medals(
    df,
    caption="Pareto Frontier",
    label="tab:pareto_frontier",
    float_format=".3f",
    largest_is_better=False,
    index_name="Method",
    columns_to_exclude=[],
    mapping_prefixes=None
) -> str:
    
    df = df.copy()
    
    df['Method'] = df['dataset_name'].map(mapping_prefixes)
    df = df.drop(columns=['dataset_name'])
    
    unmapped = df['Method'].isnull()
    if unmapped.any():
        df.loc[unmapped, 'Method'] = df.loc[unmapped, 'dataset_name']
        print(f"Warning: {unmapped.sum()} dataset_name(s) were not mapped and will use original names.")
    
    df.set_index('Method', inplace=True)
    
    if columns_to_exclude:
        df = df.drop(columns=columns_to_exclude, errors='ignore')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    ranks = pd.DataFrame(index=df.index)
    
    for col in numeric_cols:
        ranks[col] = df[col].rank(method='dense', ascending=not largest_is_better)
    
    lines = []
    # lines.append(r"\begin{table}[t!]")
    # lines.append(r"    \centering")
    # lines.append(f"    \\caption{{{caption}}}")
    # lines.append(f"    \\label{{{label}}}")
    
    column_alignment = "l" + "c" * len(df.columns)
    lines.append(f"    \\begin{{tabular}}{{{column_alignment}}}")
    lines.append(r"    \toprule")
    
    header_cols = [index_name] + [str(col).capitalize() for col in df.columns]
    lines.append("    " + " & ".join(header_cols) + r" \\")
    lines.append(r"    \midrule")
    
    for idx in df.index:
        row_vals = [str(idx)]
        for col in df.columns:
            val = df.loc[idx, col]
            # if pd.notnull(val) and col in numeric_cols:
            if pd.notnull(df[col]).all():
                try:
                    val_str = f"{val:{float_format}}"
                except Exception as e:
                    print(f"Error: {val} is not a number.")
                    print()
                    raise e

                rank_val = ranks.loc[idx, col]
                if rank_val == 1:
                    val_str = r"\cellcolor{gold!30}" + val_str
                elif rank_val == 2:
                    val_str = r"\cellcolor{silver!30}" + val_str
                elif rank_val == 3:
                    val_str = r"\cellcolor{bronze!30}" + val_str
                row_vals.append(val_str)
            else:
                row_vals.append(str(val))
        line = " & ".join(row_vals) + r" \\"
        lines.append("    " + line)
    
    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    # lines.append(r"\end{table}")
    
    latex_str = "\n".join(lines)
    return latex_str