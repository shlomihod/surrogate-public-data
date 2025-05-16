import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from snsynth.base import Synthesizer
from snsynth.utils import cdp_rho, exponential_mechanism, gaussian_noise, powerset

from mbi import (
    Dataset,
    Domain,
    estimation,
    junction_tree,
    LinearMeasurement
)

def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))

def compile_workload(workload):
    def score(cl):
        return sum(len(set(cl) & set(ax)) for ax in workload)

    return {cl: score(cl) for cl in downward_closure(workload)}

def hypothetical_model_size(domain, cliques):
    # quick approach is to make a junction tree -
    # iters=0 and get the sum of the domain sizes for the maximal cliques
    jtree, _ = junction_tree.make_junction_tree(domain, cliques)
    maximal_cliques = junction_tree.maximal_cliques(jtree)
    cells = sum(domain.size(cl) for cl in maximal_cliques)
    size_mb = cells * 8 / 2**20
    return size_mb

def filter_candidates(candidates, model, size_limit):
    ans = {}
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = (
            hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        )
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans


class AIMSynthesizerJax(Synthesizer):
    """
    AIMSynthesizerTorch -> AIMSynthesizerJax updated to use new mbi (JAX-based) code.
    """
    def __init__(
        self,
        epsilon=1.0,
        delta=1e-9,
        max_model_size=80,
        degree=2,
        num_marginals=None,
        max_cells=10000,
        rounds=None,
        verbose=False,
        max_iters=1000,
        structural_zeros={}
    ):
        super().__init__()
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.max_model_size = max_model_size
        self.degree = degree
        self.num_marginals = num_marginals
        self.max_cells = max_cells
        self.rounds = rounds
        self.verbose = verbose
        self.max_iters = max_iters

        self.structural_zeros = structural_zeros

        self.synthesizer = None
        self.num_rows = None
        self.original_column_names = None

        self.rho = 0.0 if self.delta == 0.0 else cdp_rho(self.epsilon, self.delta)

    def fit(
        self,
        data,
        *ignore,
        transformer=None,
        categorical_columns=None,
        ordinal_columns=None,
        continuous_columns=None,
        preprocessor_eps=0.0,
        nullable=False,
    ):
        # if not self.torch_backend:
        if type(data) is pd.DataFrame:
            self.original_column_names = data.columns

        train_data = self._get_train_data(
            data,
            style='cube',
            transformer=transformer,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns,
            continuous_columns=continuous_columns,
            nullable=nullable,
            preprocessor_eps=preprocessor_eps
        )

        print(train_data)
        print(type(train_data))

        if self._transformer is None:
            raise ValueError("We weren't able to fit a transformer to the data. Please check your data and try again.")

        cards = self._transformer.cardinality
        if any(c is None for c in cards):
            raise ValueError("The transformer appears to have some continuous columns. Please provide only categorical or ordinal.")

        print(self._transformer.output_width)
        colnames = ["col" + str(i) for i in range(self._transformer.output_width)]

        dimensionality = np.prod(cards)
        if self.verbose:
            print(f"Fitting with {dimensionality} dimensions")

        if len(cards) != len(colnames):
            raise ValueError("Cardinality and column names must be the same length.")

        domain = Domain(colnames, cards)
        self.num_rows = len(train_data)

        self.rho = 0.0 if self.delta == 0.0 else cdp_rho(self.epsilon, self.delta)
        if self.verbose:
            print(f"Rho: {self.rho}")

        data = pd.DataFrame(train_data, columns=colnames)
        data = Dataset(df=data, domain=domain)
        workload = self.get_workload(
            data, degree=self.degree, max_cells=self.max_cells, num_marginals=self.num_marginals
        )

        self._AIM(data, workload)

    def sample(self, samples=None):
        if samples is None:
            samples = self.num_rows
        if self.synthesizer is None:
            raise ValueError("Model has not been fit yet.")

        synth = self.synthesizer.synthetic_data(rows=samples)
        df_synth = synth.df  # a pd.DataFrame of integer-coded columns
        data_iter = df_synth.itertuples(index=False, name=None)
        return self._transformer.inverse_transform(data_iter)
    
    @staticmethod
    def get_workload(data: Dataset, degree: int, max_cells: int, num_marginals: int = None):
        workload = list(itertools.combinations(data.domain, degree))
        workload = [cl for cl in workload if data.domain.size(cl) <= max_cells]

        # workload = [(cl, 1.0) for cl in workload]
        return workload
    
    @staticmethod
    def get_workload(data: Dataset, degree: int, max_cells: int, num_marginals: int = None):
        workload = list(itertools.combinations(data.domain, degree))
        workload = [cl for cl in workload if data.domain.size(cl) <= max_cells]

        # workload = [(cl, 1.0) for cl in workload]
        return workload

    def _AIM(self, data_mbi, workload, num_synth_rows=None, initial_cliques=None):
        """
        The main iterative procedure that picks new cliques adaptively
        and runs mirror descent after each new measurement.
        """
        domain = data_mbi.domain
        n_attrs = len(domain)
        
        rounds = self.rounds or (16 * n_attrs)
        print(workload)
        candidates = compile_workload(workload)

        answers = {cl: data_mbi.project(cl).datavector() for cl in candidates}

        if not initial_cliques:
            initial_cliques = [
                cl for cl in candidates if len(cl) == 1
            ] 

        oneway = [cl for cl in candidates if len(cl) == 1]

        sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
        eps = np.sqrt(8 * 0.1 * self.rho / rounds)

        if self.verbose:
            print(f"Initial Sigma: {sigma}, Eps: {eps}")

        measurements = []
        rho_used = len(oneway) * 0.5 / sigma**2
        for cl in initial_cliques:
            x = data_mbi.project(cl).datavector()
            y = x + gaussian_noise(sigma, x.size)
            measurements.append(LinearMeasurement(y, cl, stddev=sigma))

        zeros = self.structural_zeros
        # NOTE: Haven't incorproated structural zeros back yet after refactoring
        model = estimation.mirror_descent(
                data_mbi.domain, measurements, iters=self.max_iters, callback_fn=lambda *_: None
        )

        t = 0
        terminate = False
        while not terminate:
            t += 1
            if self.rho - rho_used < 2 * (0.5 / sigma**2 + 1.0 / 8 * self.epsilon**2):
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                self.epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True

            rho_used += 1.0 / 8 * self.epsilon**2 + 0.5 / sigma**2
            print('Budget Used', rho_used, '/', self.rho)
            size_limit = self.max_model_size * rho_used / self.rho

            small_candidates = filter_candidates(candidates, model, size_limit)
            cl = self.worst_approximated(
                small_candidates, answers, model, self.epsilon, sigma
            )
            print('Measuring Clique', cl)
            n = data_mbi.domain.size(cl)
            x = data_mbi.project(cl).datavector()
            y = x + gaussian_noise(sigma, n)
            measurements.append(LinearMeasurement(y, cl, stddev=sigma))
            z = model.project(cl).datavector()

            # warm start potentials from prior round
            # TODO: check if it helps to call maximal_subsets here
            pcliques = list(set(M.clique for M in measurements))
            potentials = model.potentials.expand(pcliques)
            model = estimation.mirror_descent(
                    data_mbi.domain, measurements, iters=self.max_iters, potentials=potentials, callback_fn=lambda *_: None
            )
            w = model.project(cl).datavector()
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                print("(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma", sigma / 2)
                sigma /= 2
                self.epsilon *= 2

        if self.verbose:
            print("\n--- Finished iterative AIM procedure.  Building final synthetic data. ---")

        model = estimation.mirror_descent(
            data_mbi.domain, measurements, iters=self.max_iters
        )
        self.synthesizer = model  # store for later

    def worst_approximated(self, candidates, answers, model, eps, sigma):
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)

        max_sensitivity = max(
            sensitivity.values()
        )  # if all weights are 0, could be a problem

        print('Check weights:')
        print(errors)
        print(max_sensitivity)
        print()
        return exponential_mechanism(errors, eps, max_sensitivity)
