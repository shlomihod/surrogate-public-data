import numpy as np
import torch
from itertools import chain, combinations, product
from typing import Optional

import pandas as pd

from ydnpd.utils import metadata_to_pandera_schema


class RandomBayesianNetwork:
    def __init__(self, schema: dict, max_degree: int, alpha: float = 1.0, seed: Optional[int] = None):
        """
        Initialize RandomBayesian Network generator.

        Args:
            schema: Dictionary containing variables and their domains
            max_degree: Maximum number of parents per node
            alpha: Dirichlet concentration parameter
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.torch_gen = torch.Generator()
        if seed is not None:
            self.torch_gen.manual_seed(seed)

        self.schema = schema
        self.pandera_schema = metadata_to_pandera_schema(schema)
        self.max_degree = max_degree
        self.alpha = alpha
        # Store domains and their sizes
        self.domains = {
            node: list(values['values'].keys())
            for node, values in schema.items()
        }
        self.value_to_idx = {
            node: {val: idx for idx, val in enumerate(self.domains[node])}
            for node in schema
        }
        # Generate network structure and parameters
        self.network = self._generate_network()

    def _all_subsets_up_to_size(self, elements: list[str], max_size: int) -> list[list[str]]:
        """Generate all possible subsets of elements up to max_size."""
        subsets = list(chain.from_iterable(
            combinations(elements, r) for r in range(max_size + 1)
        ))
        return [list(subset) for subset in subsets]

    def _get_parent_configurations(self, parents: list[str]) -> list[tuple[int, ...]]:
        """Get all possible parent value combinations using indices."""
        if not parents:
            return []

        # Get number of values for each parent
        parent_domains = [range(len(self.domains[p])) for p in parents]
        # Generate all possible combinations
        return list(product(*parent_domains))

    def _generate_cpt(self, node: str, parent_configs: list[tuple[int, ...]]) -> dict[tuple[int, ...], np.ndarray]:
        """Generate CPT using Dirichlet distribution."""
        num_outcomes = len(self.domains[node])
        cpt = {}

        if not parent_configs:
            # No parents - single distribution
            cpt[()] = self.rng.dirichlet([self.alpha] * num_outcomes)
            return cpt

        # Generate distribution for each parent configuration
        for config in parent_configs:
            cpt[tuple(config)] = self.rng.dirichlet([self.alpha] * num_outcomes)

        return cpt

    def _generate_network(self) -> dict:
        """Generate random network structure and parameters."""
        variables = list(self.schema.keys())
        self.rng.shuffle(variables)  # Using numpy's rng for shuffling

        network = {
            'structure': [],
            'cpts': {}
        }

        for i, var in enumerate(variables):
            # Select random parent set
            possible_parents = variables[:i]
            all_possible_parent_sets = self._all_subsets_up_to_size(
                possible_parents,
                min(self.max_degree, len(possible_parents))
            )
            parents = all_possible_parent_sets[self.rng.integers(len(all_possible_parent_sets))]

            network['structure'].append((var, parents))

            # Generate CPT
            parent_configs = self._get_parent_configurations(parents)
            cpt = self._generate_cpt(var, parent_configs)

            network['cpts'][var] = {
                'parents': parents,
                'cpt': cpt
            }

        return network

    def sample(self, num_samples: int = 1) -> dict:
        """
        Generate samples from the Bayesian network.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Dictionary mapping variable names to arrays of sampled values
        """
        samples = {}

        for node, parents in self.network['structure']:
            node_cpt = self.network['cpts'][node]
            domain_values = self.domains[node]

            if parents:
                # For each sample, get parent indices and corresponding probabilities
                batch_probs = np.array([
                    node_cpt['cpt'][tuple(self.value_to_idx[p][samples[p][i]] for p in parents)]
                    for i in range(num_samples)
                ])
            else:
                # No parents - use unconditional distribution
                batch_probs = np.tile(node_cpt['cpt'][()], (num_samples, 1))

            # Sample from categorical distribution
            indices = np.array([
                self.rng.choice(len(domain_values), p=probs)
                for probs in batch_probs
            ])

            # Map indices to domain values
            samples[node] = np.array([domain_values[i] for i in indices])

        df = pd.DataFrame(samples)
        self.pandera_schema.validate(df)

        return df

    def print_structure(self, with_cpt: bool = False):
        """Print network structure and CPTs."""
        print("Bayesian Network Structure:")
        print("-" * 50)

        for node, parents in self.network['structure']:
            if parents:
                print(f"{node} <- {', '.join(parents)}")
            else:
                print(f"{node} (no parents)")

        if with_cpt:
            print("\nConditional Probability Tables:")
            print("-" * 50)

            for node, node_info in self.network['cpts'].items():
                print(f"\nCPT for {node}:")
                parents = node_info['parents']
                cpt = node_info['cpt']
                values_list = list(self.schema[node]['values'].values())

                if not parents:
                    probs = cpt[()]
                    for idx, name in enumerate(values_list):
                        print(f"  P({node}={name}) = {probs[idx]:.3f}")
                else:
                    for parent_config, probs in cpt.items():
                        config_str = ", ".join(
                            f"{p}={list(self.schema[p]['values'].values())[idx]}"
                            for p, idx in zip(parents, parent_config)
                        )
                        print(f"\n  Parent configuration: {config_str}")
                        for idx, name in enumerate(values_list):
                            print(f"    P({node}={name} | {config_str}) = {probs[idx]:.3f}")