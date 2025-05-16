# Do You Really Need Public Data? Surrogate Public Data for Differential Privacy on Tabular Data

Shlomi Hod\*, Lucas Rosenblatt\*, Julia Stoyanovich

## Installation

Run the bash script to set up the environment:
```bash

./setup.sh  # (on Linux)
```

Create an `.env` file at the root of this repository with the following keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `TOGETHER_API_KEY`.

## Design

The codebase performs two main tasks:

1. Generating surrogate public data (baselines and LLM-based)
2. Executing our large-scale evaluation of DP auxiliary tasks

Most of the heavy lifting and core algorithms are implemented within the `ydnpd` package which is part of this repository. The package also prepares three publicly available datasets published by others (ACS, EDAD, WE) for conducting the benchmarking (see `ydnpd.datasets`).

Jupyter Notebooks (at the root of the repository) are used to orchestrate the execution of surrogate data generation and evaluation at scale, and produce plots and tables.

## Surrogate Data Generation

The baseline datasets were created and included as part of the `ydnpd.datasets` sub-package. The `generate-csv.ipynb` and `generate-agent.ipynb` notebooks are used to generate the LLM-based surrogate datasets, and the specific generated datasets used in the paper are saved in the `llm_datasets` directory.

## Evaluation

Following the structure of the evaluation framework, there are three DP auxiliary tasks to be considered, grouped into two:

- Task 1: Classification pretraining
- Synthetic data generation harness:
  - Task 2: Hyperparameter tuning
  - Task 3: Privacy/utility estimation

Notebooks match the name for each group.

Complete results are available in the results folder

## Logs

Raw evaluation results and LLM/Agent-based method traces were collected via Weights & Biases: [agent traces](https://wandb.ai/shlomihod/ydnpd-data_gen_agent/weave), [pretraining raw results](https://wandb.ai/shlomihod/ydnpd-dp-ft), [harness raw results](https://wandb.ai/shlomihod/ydnpd-harness).

## Additional Notebooks

### Similarity Analysis

The `similarity.ipynb` provides similarity analysis between private and (traditional and surrogate) public datasets and among all datasets.

### Memorization Test

The `memorization.ipynb` provides the full execution of Brodt et al. methods on the three raw datasets as they are available on the internet (ACS, EDAD, WE) and the three LLMs used in this work (Claude 3.5 Sonnet, GPT-4o, Llama 3.3 70B).