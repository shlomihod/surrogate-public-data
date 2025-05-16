import os
import multiprocessing
import argparse
from pathlib import Path
import tempfile


PROJECT_NAME = "ydnpd-dp-ft"
NUM_RUNS = 10


def get_sweep_config(dataset_family, public_dataaset_pointers,
                     num_runs=NUM_RUNS, save_artifact=False,
                     subsampling=False):
    from ydnpd import ALL_EXPERIMENTS
    from ydnpd.harness.config import EPSILONS

    parameters = {
        "run_num": {"values": list(range(num_runs))},
        "save_artifact": {"value": save_artifact},
        "dp_num_epochs": {"value": 20},
        "dp_batch_size": {"value": 128},
        "dp_lr": {"values": [3e-3, 3e-4]},
        "epsilon": {"values": EPSILONS},
        "private_data_pointer": {"value": ALL_EXPERIMENTS[dataset_family].test_name},
    }

    if subsampling:
        parameters["subsampling"] = {"values": [0.2, 0.5]}

    if public_dataaset_pointers:
        parameters |= {
            "pre_num_epochs": {"values": [1 ,9]},
            "pre_batch_size": {"values": [32, 128]},
            "pre_lr": {"values": [3e-4, 3e-5]},
            "public_data_pointer": {"values": public_dataaset_pointers},
        }
    else:
        parameters["public_data_pointer"] = {"value": ""}

    return {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "dp/private.test/auc"},
        "parameters": parameters
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run wandb sweep agents with GPU/CPU distribution',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='''
    Example usage:
    python script.py --sweep_id 7i09oqiw --num_gpus 4 --num_cpus 24  # Uses 4 GPUs + 20 CPUs
    python script.py --sweep_id abc123 --num_gpus 0 --num_cpus 4     # Uses 4 CPUs only
    ''')
    parser.add_argument('--sweep_id', type=str, required=True, help='Weights & Biases sweep ID')
    parser.add_argument('--num_gpus', type=int, required=True, help='Number of GPU processes')
    parser.add_argument('--num_cpus', type=int, required=True, help='Total number of processes')
    return parser.parse_args()


def runner():
    import wandb
    from ydnpd.pretraining.trainer import TransformerTrainer, ModelConfig, PreTrainConfig
    from ydnpd.pretraining.utils import set_strict_reproducibility_by_config

    run = wandb.init(project=PROJECT_NAME)
    print(wandb.config)
    set_strict_reproducibility_by_config(wandb.config)

    with tempfile.TemporaryDirectory() as temp_dir:

        if wandb.config.public_data_pointer:
            pretrain_config = PreTrainConfig(
                num_epochs=wandb.config.pre_num_epochs,
                batch_size=wandb.config.pre_batch_size,
                lr=wandb.config.pre_lr)
            public_data_pointer = wandb.config.public_data_pointer
        else:
            pretrain_config = None
            public_data_pointer = None

        results = TransformerTrainer.train_and_evaluate(
            config=ModelConfig(
                num_epochs=wandb.config.dp_num_epochs,
                batch_size=wandb.config.dp_batch_size,
                lr=wandb.config.dp_lr,
                epsilon=wandb.config.epsilon
            ),
            pretrain_config=pretrain_config,
            public_data_pointer=public_data_pointer,
            private_data_pointer=wandb.config.private_data_pointer,
            subsampling=None if not wandb.config.subsampling else float(wandb.config.subsampling),
            save_path=temp_dir,
        )
        wandb.log(results)

        if wandb.config.save_artifact:
            for model_path in Path(temp_dir).glob("*.pkl"):
                artifact = wandb.Artifact(model_path.stem, type="model")
                artifact.add_file(str(model_path))
                run.log_artifact(artifact)

    run.finish()


def run_agent(cuda_device, num_gpus, runner_fn, sweep_id):
    if cuda_device < num_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    import wandb
    wandb.agent(sweep_id, function=runner_fn, project=PROJECT_NAME)


if __name__ == '__main__':
    args = parse_args()
    processes = []
    for device in range(args.num_cpus):
        p = multiprocessing.Process(target=run_agent, args=(device, args.num_gpus, runner, args.sweep_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
