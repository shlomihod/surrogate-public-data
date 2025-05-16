# NOTE: This file contains redundancies, will be cleaned up by:
# 1. Letting FactoredInferenceTorch wrap FactoredInference instead of redefining the class
# 2. Letting AIMSynthesizerTorch wrap AIMSynthesizer instead of redefining the class
# Will be the subject of an snsynth PR at a later date.

import numpy as np
import pandas as pd
import torch
import copy
import itertools
from collections import defaultdict
from functools import partial

from mbi import Domain, GraphicalModel, CliqueVector, callbacks
try:
    from mbi import FactoredInference, Dataset
except ImportError:
    print("Please install mbi with:\n   pip install git+https://github.com/ryan112358/private-pgm.git@01f02f17eba440f4e76c1d06fa5ee9eed0bd2bca")

from scipy.sparse.linalg import LinearOperator, eigsh, lsmr, aslinearoperator
from scipy import optimize, sparse
from snsynth.base import Synthesizer
from snsynth.utils import cdp_rho, exponential_mechanism, gaussian_noise, powerset

class FactoredInferenceTorch:
    def __init__(
        self,
        domain,
        backend="numpy",
        structural_zeros={},
        metric="L2",
        log=False,
        iters=1000,
        warm_start=False,
        elim_order=None,
        verbose=False
    ):
        """
        Class for learning a GraphicalModel from  noisy measurements on a data distribution

        :param domain: The domain information (A Domain object)
        :param backend: numpy or torch backend
        :param structural_zeros: An encoding of the known (structural) zeros in the distribution.
            Specified as a dictionary where
                - each key is a subset of attributes of size r
                - each value is a list of r-tuples corresponding to impossible attribute settings
        :param metric: The optimization metric.  May be L1, L2 or a custom callable function
            - custom callable function must consume the marginals and produce the loss and gradient
            - see FactoredInference._marginal_loss for more information
        :param log: flag to log iterations of optimization
        :param iters: number of iterations to optimize for
        :param warm_start: initialize new model or reuse last model when calling infer multiple times
        :param elim_order: an elimination order for the JunctionTree algorithm
            - Elimination order will impact the efficiency by not correctness.
              By default, a greedy elimination order is used
        """
        self.domain = domain
        self.backend = backend
        self.metric = metric
        self.log = log
        self.iters = iters
        self.warm_start = warm_start
        self.history = []
        self.elim_order = elim_order
        self.verbose = verbose
        if backend == "torch":
            from mbi.torch_factor import Factor
            from mbi import Factor as NumpyFactor

            self.Factor = Factor
            self.NumpyFactor = NumpyFactor
        else:
            from mbi import Factor

            self.Factor = Factor

        self.structural_zeros = CliqueVector({})
        for cl in structural_zeros:
            dom = self.domain.project(cl)
            fact = structural_zeros[cl]
            self.structural_zeros[cl] = self.Factor.active(dom, fact)

    def estimate(
        self, measurements, total=None, engine="MD", callback=None, options={}
    ):
        """
        Estimate a GraphicalModel from the given measurements

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param engine: the optimization algorithm to use, options include:
            MD - Mirror Descent with armijo line search
            RDA - Regularized Dual Averaging
            IG - Interior Gradient
        :param callback: a function to be called after each iteration of optimization
        :param options: solver specific options passed as a dictionary
            { param_name : param_value }

        :return model: A GraphicalModel that best matches the measurements taken
        """
        measurements = self.fix_measurements(measurements)
        options["callback"] = callback
        if callback is None and self.log:
            options["callback"] = callbacks.Logger(self)
        if engine == "MD":
            self.mirror_descent(measurements, total, **options)
        elif engine == "RDA":
            self.dual_averaging(measurements, total, **options)
        elif engine == "IG":
            self.interior_gradient(measurements, total, **options)
        return self.model

    def fix_measurements(self, measurements):
        assert type(measurements) is list, (
            "measurements must be a list, given " + measurements
        )
        assert all(
            len(m) == 4 for m in measurements
        ), "each measurement must be a 4-tuple (Q, y, noise,proj)"
        ans = []
        for Q, y, noise, proj in measurements:
            assert (
                Q is None or Q.shape[0] == y.size
            ), "shapes of Q and y are not compatible"
            if type(proj) is list:
                proj = tuple(proj)
            if type(proj) is not tuple:
                proj = (proj,)
            if Q is None:
                Q = sparse.eye(self.domain.size(proj))
            assert np.isscalar(noise), "noise must be a real value, given " + str(noise)
            assert all(a in self.domain for a in proj), (
                str(proj) + " not contained in domain"
            )
            assert Q.shape[1] == self.domain.size(
                proj
            ), "shapes of Q and proj are not compatible"
            ans.append((Q, y, noise, proj))
        return ans

    def interior_gradient(
        self, measurements, total, lipschitz=None, c=1, sigma=1, callback=None
    ):
        """ Use the interior gradient algorithm to estimate the GraphicalModel
            See https://epubs.siam.org/doi/pdf/10.1137/S1052623403427823 for more information

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param lipschitz: the Lipchitz constant of grad L(mu)
            - automatically calculated for metric=L2
            - doesn't exist for metric=L1
            - must be supplied for custom callable metrics
        :param c, sigma: parameters of the algorithm
        :param callback: a function to be called after each iteration of optimization
        """
        assert self.metric != "L1", "dual_averaging cannot be used with metric=L1"
        assert (
            not callable(self.metric) or lipschitz is not None
        ), "lipschitz constant must be supplied"
        self._setup(measurements, total)
        # what are c and sigma?  For now using 1
        model = self.model
        domain, cliques, total = model.domain, model.cliques, model.total
        L = self._lipschitz(measurements) if lipschitz is None else lipschitz
        if self.log:
            print("Lipchitz constant:", L)

        theta = model.potentials
        x = y = z = model.belief_propagation(theta)
        c0 = c
        l = sigma / L
        for k in range(1, self.iters + 1):
            a = (np.sqrt((c * l) ** 2 + 4 * c * l) - l * c) / 2
            y = (1 - a) * x + a * z
            c *= 1 - a
            _, g = self._marginal_loss(y)
            theta = theta - a / c / total * g
            z = model.belief_propagation(theta)
            x = (1 - a) * x + a * z
            if callback is not None:
                callback(x)

        model.marginals = x
        model.potentials = model.mle(x)

    def dual_averaging(self, measurements, total=None, lipschitz=None, callback=None):
        """ Use the regularized dual averaging algorithm to estimate the GraphicalModel
            See https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/xiao10JMLR.pdf

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param lipschitz: the Lipchitz constant of grad L(mu)
            - automatically calculated for metric=L2
            - doesn't exist for metric=L1
            - must be supplied for custom callable metrics
        :param callback: a function to be called after each iteration of optimization
        """
        assert self.metric != "L1", "dual_averaging cannot be used with metric=L1"
        assert (
            not callable(self.metric) or lipschitz is not None
        ), "lipschitz constant must be supplied"
        self._setup(measurements, total)
        model = self.model
        domain, cliques, total = model.domain, model.cliques, model.total
        L = self._lipschitz(measurements) if lipschitz is None else lipschitz
        print("Lipchitz constant:", L)
        if L == 0:
            return

        theta = model.potentials
        gbar = CliqueVector(
            {cl: self.Factor.zeros(domain.project(cl)) for cl in cliques}
        )
        w = v = model.belief_propagation(theta)
        beta = 0

        for t in range(1, self.iters + 1):
            c = 2.0 / (t + 1)
            u = (1 - c) * w + c * v
            _, g = self._marginal_loss(u)  # not interested in loss of this query point
            gbar = (1 - c) * gbar + c * g
            theta = -t * (t + 1) / (4 * L + beta) / self.model.total * gbar
            v = model.belief_propagation(theta)
            w = (1 - c) * w + c * v

            if callback is not None:
                callback(w)

        model.marginals = w
        model.potentials = model.mle(w)

    def mirror_descent(self, measurements, total=None, stepsize=None, callback=None):
        """ Use the mirror descent algorithm to estimate the GraphicalModel
            See https://web.iem.technion.ac.il/images/user-files/becka/papers/3.pdf

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param stepsize: The step size function for the optimization (None or scalar or function)
            if None, will perform line search at each iteration (requires smooth objective)
            if scalar, will use constant step size
            if function, will be called with the iteration number
        :param total: The total number of records (if known)
        :param callback: a function to be called after each iteration of optimization
        """
        assert not (
            self.metric == "L1" and stepsize is None
        ), "loss function not smooth, cannot use line search (specify stepsize)"

        self._setup(measurements, total)
        model = self.model
        cliques, theta = model.cliques, model.potentials
        mu = model.belief_propagation(theta)
        ans = self._marginal_loss(mu)
        if ans[0] == 0:
            return ans[0]

        nols = stepsize is not None
        if np.isscalar(stepsize):
            alpha = float(stepsize)
            stepsize = lambda t: alpha
        if stepsize is None:
            alpha = 1.0 / self.model.total ** 2
            stepsize = lambda t: 2.0 * alpha

        for t in range(1, self.iters + 1):
            if callback is not None:
                callback(mu)
            omega, nu = theta, mu
            curr_loss, dL = ans
            # print('Gradient Norm', np.sqrt(dL.dot(dL)))
            alpha = stepsize(t)

            if self.backend == "torch":
                if self.verbose:
                    print(omega)
                    print(alpha)
                    print(dL)
                    print()
                    print(dL[('col0',)].values)
                    print(copy.copy(dL[('col0',)].values).cpu().detach().numpy())
                    print()
                dL_temp = CliqueVector({cl: self.NumpyFactor(domain=copy.copy(dL[cl].domain),
                                                            values=copy.copy(dL[cl].values).cpu().detach().numpy())
                                            for cl in cliques})
            else:
                dL_temp = dL

            if self.verbose:
                print(dL_temp[('col0',)].values)
            for i in range(25):
                theta = omega - alpha * dL_temp

                mu = model.belief_propagation(theta)
                ans = self._marginal_loss(mu)
                if nols or curr_loss - ans[0] >= 0.5 * alpha * dL_temp.dot(nu - mu):
                    break
                alpha *= 0.5



        model.potentials = theta
        model.marginals = mu

        return ans[0]

    def _marginal_loss(self, marginals, metric=None):
        """ Compute the loss and gradient for a given dictionary of marginals

        :param marginals: A dictionary with keys as projections and values as Factors
        :return loss: the loss value
        :return grad: A dictionary with gradient for each marginal
        """
        if metric is None:
            metric = self.metric

        if callable(metric):
            return metric(marginals)

        loss = 0.0
        gradient = {}

        for cl in marginals:
            mu = marginals[cl]
            gradient[cl] = self.Factor.zeros(mu.domain)
            for Q, y, noise, proj in self.groups[cl]:
                c = 1.0 / noise
                mu2 = mu.project(proj)
                x = mu2.datavector()

                if self.backend == "torch":
                    c = torch.tensor(c, dtype=torch.float32, device=y.device)
                    if isinstance(Q, np.ndarray):
                        Q = torch.tensor(Q, dtype=torch.float32, device=y.device)
                    elif sparse.issparse(Q):
                        Q = Q.tocoo()
                        idx = torch.LongTensor([Q.row, Q.col])
                        vals = torch.FloatTensor(Q.data).to(y.device)
                        Q = torch.sparse.FloatTensor(idx, vals).to(y.device)
                    x = torch.tensor(x, dtype=torch.float32, device=y.device)

                diff = c * (Q @ x - y)

                if metric == "L1":
                    loss += abs(diff).sum()
                    sign = diff.sign() if hasattr(diff, "sign") else np.sign(diff)
                    grad = c * (Q.T @ sign)
                else:
                    loss += 0.5 * (diff @ diff).sum()
                    grad = c * (Q.T @ diff)

                gradient[cl] += self.Factor(mu2.domain, grad)

        return float(loss), CliqueVector(gradient)

    def _setup(self, measurements, total):
        """ Perform necessary setup for running estimation algorithms

        1. If total is None, find the minimum variance unbiased estimate for total and use that
        2. Construct the GraphicalModel
            * If there are structural_zeros in the distribution, initialize factors appropriately
        3. Pre-process measurements into groups so that _marginal_loss may be evaluated efficiently
        """
        if total is None:
            # find the minimum variance estimate of the total given the measurements
            variances = np.array([])
            estimates = np.array([])
            for Q, y, noise, proj in measurements:
                o = np.ones(Q.shape[1])
                v = lsmr(Q.T, o, atol=0, btol=0)[0]
                if np.allclose(Q.T.dot(v), o):
                    variances = np.append(variances, noise ** 2 * np.dot(v, v))
                    estimates = np.append(estimates, np.dot(v, y))
            if estimates.size == 0:
                total = 1
            else:
                variance = 1.0 / np.sum(1.0 / variances)
                estimate = variance * np.sum(estimates / variances)
                total = max(1, estimate)

        # if not self.warm_start or not hasattr(self, 'model'):
        # initialize the model and parameters
        cliques = [m[3] for m in measurements]
        if self.structural_zeros is not None:
            cliques += list(self.structural_zeros.keys())

        model = GraphicalModel(
            self.domain, cliques, total, elimination_order=self.elim_order
        )

        model.potentials = CliqueVector.zeros(self.domain, model.cliques)
        model.potentials.combine(self.structural_zeros)
        if self.warm_start and hasattr(self, "model"):
            model.potentials.combine(self.model.potentials)
        self.model = model

        # group the measurements into model cliques
        cliques = self.model.cliques
        # self.groups = { cl : [] for cl in cliques }
        self.groups = defaultdict(lambda: [])
        for Q, y, noise, proj in measurements:
            if self.backend == "torch":
                import torch

                device = self.Factor.device
                y = torch.tensor(y, dtype=torch.float32, device=device)
                if isinstance(Q, np.ndarray):
                    Q = torch.tensor(Q, dtype=torch.float32, device=device)
                elif sparse.issparse(Q):
                    Q = Q.tocoo()
                    idx = torch.LongTensor([Q.row, Q.col])
                    vals = torch.FloatTensor(Q.data)
                    Q = torch.sparse.FloatTensor(idx, vals).to(device)

                # else Q is a Linear Operator, must be compatible with torch
            m = (Q, y, noise, proj)
            for cl in sorted(cliques, key=model.domain.size):
                # (Q, y, noise, proj) tuple
                if set(proj) <= set(cl):
                    self.groups[cl].append(m)
                    break

    def _lipschitz(self, measurements):
        """ compute lipschitz constant for L2 loss

            Note: must be called after _setup
        """
        eigs = {cl: 0.0 for cl in self.model.cliques}
        for Q, _, noise, proj in measurements:
            for cl in self.model.cliques:
                if set(proj) <= set(cl):
                    n = self.domain.size(cl)
                    p = self.domain.size(proj)
                    Q = aslinearoperator(Q)
                    Q.dtype = np.dtype(Q.dtype)
                    eig = eigsh(Q.H * Q, 1)[0][0]
                    eigs[cl] += eig * n / p / noise ** 2
                    break
        return max(eigs.values())

    def infer(self, measurements, total=None, engine="MD", callback=None, options={}):
        import warnings

        message = "Function infer is deprecated.  Please use estimate instead."
        warnings.warn(message, DeprecationWarning)
        return self.estimate(measurements, total, engine, callback, options)
    

prng = np.random


class Identity(sparse.linalg.LinearOperator):
    def __init__(self, n, torch_backend=False):
        self.shape = (n,n)
        if torch_backend:
            self.dtype = torch.float64
        else:
          self.dtype = np.float64
    def _matmat(self, X):
        return X
    def __matmul__(self, X):
        return X
    def _transpose(self):
        return self
    def _adjoint(self):
        return self

def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def hypothetical_model_size(domain, cliques):
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2 ** 20


def compile_workload(workload):
    def score(cl):
        return sum(len(set(cl) & set(ax)) for ax in workload)

    return {cl: score(cl) for cl in downward_closure(workload)}


def filter_candidates(candidates, model, size_limit):
    ans = {}
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans


class AIMSynthesizerTorch(Synthesizer):
    """AIM: An Adaptive and Iterative Mechanism

    :param epsilon: privacy budget for the synthesizer
    :type epsilon: float
    :param delta: privacy parameter.  Should be small, in the range of 1/(n * sqrt(n))
    :type delta: float
    :param verbose: print diagnostic information during processing
    :type verbose: bool

    Based on the code available in:
    https://github.com/ryan112358/private-pgm/blob/master/mechanisms/aim.py
    """

    def __init__(self, epsilon=1., delta=1e-9, max_model_size=80, degree=2, num_marginals=None, max_cells: int = 10000,
                 rounds=None, torch_backend=False, verbose=False):
        if isinstance(epsilon, int):
            epsilon = float(epsilon)
        self.rounds = rounds
        self.max_model_size = max_model_size
        self.max_cells = max_cells
        self.degree = degree
        self.num_marginals = num_marginals
        self.verbose = verbose
        self.epsilon = epsilon
        self.delta = delta
        self.synthesizer = None
        self.num_rows = None
        self.original_column_names = None
        self.torch_backend = torch_backend

    def fit(
            self,
            data,
            *ignore,
            transformer=None,
            categorical_columns=[],
            ordinal_columns=[],
            continuous_columns=[],
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

        self.AIM(data, workload)

    def sample(self, samples=None):
        if samples is None:
            samples = self.num_rows
        data = self.synthesizer.synthetic_data(rows=samples)
        data_iter = [tuple([c for c in t[1:]]) for t in data.df.itertuples()]
        if self.torch_backend:
            return self._transformer.inverse_transform(data_iter)
        return self._transformer.inverse_transform(data_iter)

    @staticmethod
    def get_workload(data: Dataset, degree: int, max_cells: int, num_marginals: int = None):
        workload = list(itertools.combinations(data.domain, degree))
        workload = [cl for cl in workload if data.domain.size(cl) <= max_cells]
        if num_marginals is not None:
            workload = [workload[i] for i in prng.choice(len(workload), num_marginals, replace=False)]

        # workload = [(cl, 1.0) for cl in workload]
        return workload

    def _worst_approximated(self, candidates, answers, model, eps, sigma):
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)

        max_sensitivity = max(sensitivity.values())  # if all weights are 0, could be a problem
        print('Check weights:')
        print(errors)
        print(max_sensitivity)
        print()
        return exponential_mechanism(errors, eps, max_sensitivity)

    def AIM(self, data, workload):
        rounds = self.rounds or 16 * len(data.domain)
        # workload = [cl for cl, _ in W]
        candidates = compile_workload(workload)
        answers = {cl: data.project(cl).datavector() for cl in candidates}

        oneway = [cl for cl in candidates if len(cl) == 1]

        sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
        epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)

        measurements = []
        print('Initial Sigma', sigma)
        rho_used = len(oneway) * 0.5 / sigma ** 2
        for cl in oneway:
            x = data.project(cl).datavector()
            y = x + gaussian_noise(sigma, x.size)
            I = Identity(y.size)
            measurements.append((I, y, sigma, cl))

        if self.torch_backend:
            engine = FactoredInferenceTorch(domain=data.domain, iters=1000, warm_start=True, backend="torch", verbose=False)
        else:
            engine = FactoredInference(data.domain, iters=1000, warm_start=True)

        Q = measurements[0][0]
        y = measurements[0][1]

        print(Q.shape[0])
        print(y.size)
        model = engine.estimate(measurements=measurements)#, engine="RDA")

        t = 0
        terminate = False
        while not terminate:
            t += 1
            if self.rho - rho_used < 2 * (0.5 / sigma ** 2 + 1.0 / 8 * epsilon ** 2):
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                print('Rho check')
                print(self.rho)
                print(rho_used)
                print(remaining)
                assert remaining > 0
                print()
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True

            rho_used += 1.0 / 8 * epsilon ** 2 + 0.5 / sigma ** 2
            size_limit = self.max_model_size * rho_used / self.rho

            small_candidates = filter_candidates(candidates, model, size_limit)
            cl = self._worst_approximated(small_candidates, answers, model, epsilon, sigma)

            n = data.domain.size(cl)
            Q = Identity(n)
            x = data.project(cl).datavector()
            y = x + gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))
            z = model.project(cl).datavector()

            model = engine.estimate(measurements=measurements)
            w = model.project(cl).datavector()
            if self.verbose:
                print('Selected', cl, 'Size', n, 'Budget Used', rho_used / self.rho)
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                if self.verbose:
                    print('(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma', sigma / 2)
                sigma /= 2
                epsilon *= 2

        engine.iters = 2500
        model = engine.estimate(measurements=measurements)

        if self.verbose:
            print("Estimating marginals")

        self.synthesizer = model

    def get_errors(self, data: Dataset, workload):
        errors = []
        for proj, wgt in workload:
            X = data.project(proj).datavector()
            Y = self.synthesizer.project(proj).datavector()
            e = 0.5 * wgt * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
            errors.append(e)
        print('Average Error: ', np.mean(errors))
        return errors