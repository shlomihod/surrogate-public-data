"""
Grid Search Analysis Tool for DP Tasks

This module provides utilities for analyzing the completeness of grid search experiments
for differential privacy tasks. It helps identify missing configurations and verify
run counts across different experimental configurations.
"""

from collections import defaultdict
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
import ydnpd


@dataclass
class ConfigStats:
    """Statistics about configurations for a synth_name/dataset pair."""
    total_configs: int
    irregular_configs: Dict[Tuple, int]
    expected_configs: int
    missing_configs: Set[Tuple]


def format_config(epsilon: float, hparams: Tuple) -> str:
    """Formats a configuration for printing."""
    return f"epsilon: {epsilon}, hparams: dict({hparams})"


def analyze_experiment_pair(
    results: List[Dict[str, Any]],
    task: 'UtilityTask'
) -> ConfigStats:
    """
    Analyzes configurations for a single synth_name/dataset_name pair.

    Args:
        results: List of result dictionaries
        task: UtilityTask object defining the expected configuration space

    Returns:
        ConfigStats object with analysis results
    """
    # Count runs per configuration
    config_runs = defaultdict(int)
    for r in results:
        if r['synth_name'] == task.synth_name and r['dataset_name'] == task.dataset_name:
            key = (r['epsilon'], tuple(sorted(r['hparams'].items())))
            config_runs[key] += 1

    # Find irregular run counts
    irregular_configs = {k: v for k, v in config_runs.items() if v != task.num_runs}

    # Get expected configurations from task
    expected_configs = {
        (eps, tuple(sorted(hparams.items())))
        for eps in task.epsilons
        for hparams in task.hparam_space
    }

    # Find missing configurations
    existing_configs = set(config_runs.keys())
    missing_configs = expected_configs - existing_configs

    return ConfigStats(
        total_configs=len(config_runs),
        irregular_configs=irregular_configs,
        expected_configs=len(expected_configs),
        missing_configs=missing_configs
    )


def analyze_grid_search_completeness(
    results: List[Dict[str, Any]],
    additional_datasets: List[str] = None
) -> None:
    """
    Analyzes grid search completeness and prints both analysis report and code for missing configurations.

    Args:
        results: List of result dictionaries
        additional_datasets: List of additional dataset names to include
    """
    tasks = ydnpd.span_utility_tasks(additional_datasets)

    print("=== Grid Search Analysis Report ===\n")

    total_results = len(results)
    total_expected = sum(task.size() for task in tasks)

    # Collect missing configurations for code generation
    missing_by_dataset = {}

    # Print analysis for each task
    for task in sorted(tasks, key=lambda t: (t.synth_name, t.dataset_name)):
        stats = analyze_experiment_pair(results, task)

        print(f"\nAnalysis for {task.synth_name} & {task.dataset_name}:")
        print(f"Total configurations found: {stats.total_configs}")
        print(f"Expected configurations: {stats.expected_configs}")

        if stats.irregular_configs:
            print("\nConfigurations with irregular run counts:")
            for config, count in sorted(stats.irregular_configs.items()):
                print(f"{format_config(*config)}, runs: {count}")
        else:
            print(f"All configurations have exactly {task.num_runs} runs")

        if stats.missing_configs:
            print("\nMissing configurations:")
            for config in sorted(stats.missing_configs):
                print(format_config(*config))
            missing_by_dataset[(task.synth_name, task.dataset_name)] = stats.missing_configs

    print("\n=== Summary ===")
    print(f"Total results: {total_results}")
    print(f"Expected total: {total_expected}")
    print(f"Missing runs: {total_expected - total_results}")

    if missing_by_dataset:
        print("\n=== Generated Code for Missing Configurations ===\n")
        code_parts = ["from ydnpd.harness.tasks import UtilityTask\n"]

        for (synth_name, dataset_name), configs in sorted(missing_by_dataset.items()):
            # Extract unique parameters
            epsilons = sorted({eps for eps, _ in configs})
            all_hparams = [dict(hparams) for _, hparams in configs]

            # Extract unique values for each hparam
            hparam_dims = {}
            for param in all_hparams[0].keys():
                hparam_dims[param] = sorted({h[param] for h in all_hparams})

            # Generate task code
            task_code = f"""# Task for {synth_name} & {dataset_name}
{dataset_name.replace('/', '_').replace('-', '_')}_task = ydnpd.UtilityTask(
    dataset_pointer="{dataset_name}",
    epsilons={epsilons},
    synth_name="{synth_name}",
    hparam_dims={hparam_dims},
    num_runs=5,
    verbose=True,
    with_wandb=True,
    evaluation_kwargs=ydnpd.harness.config.EVALUATION_KWARGS,
    wandb_kwargs={{"group": "TODO"}}

)"""
            code_parts.append(task_code)

        # Add list of all tasks
        task_names = [f"{dataset_name.replace('/', '_').replace('-', '_')}_task"
                    for synth_name, dataset_name in missing_by_dataset.keys()]
        code_parts.append("\n# List of all missing tasks")
        code_parts.append(f"missing_tasks = [{', '.join(task_names)}]")

        print('\n'.join(code_parts))
    else:
        print("\nNo missing configurations found - no code generation needed.")
