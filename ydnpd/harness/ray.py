from typing import Optional

import ray

import ydnpd


def span_utility_tasks(additional_datasets: Optional[list[tuple[str, str]]] = None,
                       task_kwargs: Optional[dict] = None):

    if task_kwargs is None:
        task_kwargs = {}

    dataset_pointers = list(ydnpd.harness.config.DATASET_NAMES)
    if additional_datasets:
        dataset_pointers.extend(additional_datasets)

    return [
        ydnpd.harness.UtilityTask(
            dataset_pointer=dataset_pointer,
            epsilons=ydnpd.harness.config.EPSILONS,
            synth_name=synth_name,
            hparam_dims=ydnpd.harness.config.HPARAMS_DIMS[synth_name],
            num_runs=ydnpd.harness.config.NUM_RUNS,
            evaluation_kwargs=ydnpd.harness.config.EVALUATION_KWARGS,
            **task_kwargs
        )
        for synth_name in ydnpd.harness.config.SYNTHESIZERS
        for dataset_pointer in dataset_pointers
    ]


def span_utility_ray_tasks(additional_datasets: Optional[list[tuple[str, str]]] = None,
                           **task_kwargs):

    def task_execute_wrapper(task):
        def function():
            return task.execute()

        return function

    return [
        ray.remote(task_execute_wrapper(task))
        .options(num_gpus=(1 if task.synth_name in ("aim_jax", "patectgan", "gem") else 0))
        .remote()
        for task in span_utility_tasks(additional_datasets, task_kwargs)
    ]
