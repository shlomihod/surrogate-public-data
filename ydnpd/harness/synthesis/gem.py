import pandas as pd
import numpy as np
from snsynth.transform.table import TableTransformer
from ydnpd.harness.synthesis.src.utils import Dataset, Domain
from ydnpd.harness.synthesis.src.qm.qm import KWayMarginalQMTorch
from ydnpd.harness.synthesis.src.syndata import NeuralNetworkGenerator
from ydnpd.harness.synthesis.src.algo import IterAlgoSingleGEM as GEM
from ydnpd.harness.synthesis.src.utils import get_rand_workloads
import torch
from snsynth.base import Synthesizer

class GEMSynthesizer(Synthesizer):
    def __init__(self, epsilon: float, 
                 slide_range: bool = False, 
                 thresh=0.05, 
                 k=3,
                 T=100,
                 alpha=0.67,
                 lr=1e-4,
                 ema_weights_beta=0.9,
                 recycle=True,
                 synth_kwargs=dict(), 
                 verbose=False):
        self.synth_kwargs = synth_kwargs
        self.verbose = verbose
        self.k = k
        self.T = T
        self.recycle = recycle
        self.thresh = thresh
        self.epsilon = epsilon
        self.slide_range = slide_range

        self.alpha = alpha
        self.lr = lr
        self.ema_weights_beta = ema_weights_beta

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def _slide_range(self, df):
        if self.slide_range:
            df, self.range_transform = self.slide_range_forward(df)
        return df

    def _unslide_range(self, df):
        if self.slide_range and self.range_transform is None:
            raise ValueError("Must fit synthesizer before sampling.")
        if self.slide_range:
            df = self.slide_range_backward(df, self.range_transform)
        return df

    def _categorical_continuous(self, df):
        # NOTE: return categorical/ordinal columns and continuous
        # This is slightly hacky, but should be fine.
        categorical = []
        continuous = []
        for col in df.columns:
            if (float(df[col].nunique()) / df[col].count()) < self.thresh:
                categorical.append(col)
            else:
                continuous.append(col)
        return {"categorical": categorical, "continuous": continuous}

    @staticmethod
    def slide_range_forward(df):
        transform = {}
        for c in df.columns:
            if min(df[c]) > 0:
                transform[c] = min(df[c])
                df[c] = df[c] - min(df[c])
        return df, transform

    @staticmethod
    def slide_range_backward(df, transform) -> pd.DataFrame:
        for c in df.columns:
            if c in transform:
                df[c] = df[c] + transform[c]
        return df

    def _get_train_data(self, data, *ignore, style, transformer, categorical_columns, ordinal_columns, continuous_columns, nullable, preprocessor_eps):
        if transformer is None or isinstance(transformer, dict):
            self._transformer = TableTransformer.create(data, style=style,
                categorical_columns=categorical_columns,
                continuous_columns=continuous_columns,
                ordinal_columns=ordinal_columns,
                nullable=nullable,
                constraints=transformer)
        elif isinstance(transformer, TableTransformer):
            self._transformer = transformer
        else:
            raise ValueError("transformer must be a TableTransformer object, a dictionary or None.")
        if not self._transformer.fit_complete:
            if self._transformer.needs_epsilon and (preprocessor_eps is None or preprocessor_eps == 0.0):
                raise ValueError("Transformer needs some epsilon to infer bounds.  If you know the bounds, pass them in to save budget.  Otherwise, set preprocessor_eps to a value > 0.0 and less than the training epsilon.  Preprocessing budget will be subtracted from training budget.")
            self._transformer.fit(
                data,
                epsilon=preprocessor_eps
            )
            eps_spent, _ = self._transformer.odometer.spent
            if eps_spent > 0.0:
                self.epsilon -= eps_spent
                print(f"Spent {eps_spent} epsilon on preprocessor, leaving {self.epsilon} for training")
                if self.epsilon < 10E-3:
                    raise ValueError("Epsilon remaining is too small!")
        train_data = self._transformer.transform(data)
        return train_data

    def fit(self, 
            df: pd.DataFrame,
            *ignore,
            transformer=None,
            categorical_columns=[],
            ordinal_columns=[],
            continuous_columns=[],
            preprocessor_eps=0.0,
            nullable=False,):
        
        if type(df) is pd.DataFrame:
            self.original_column_names = df.columns
        
        categorical_check = (len(self._categorical_continuous(df)['categorical']) == len(list(df.columns)))
        if not categorical_check:
            raise ValueError('Please make sure that GEM gets categorical/ordinal\
                features only. If you are sure you only passed categorical, \
                increase the `thresh` parameter.')
        df = self._slide_range(df)

        train_data = self._get_train_data(
            df,
            style='cube',
            transformer=transformer,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns,
            continuous_columns=continuous_columns,
            nullable=nullable,
            preprocessor_eps=preprocessor_eps
        )

        if self._transformer is None:
            raise ValueError("We weren't able to fit a transformer to the data. Please check your data and try again.")

        cards = self._transformer.cardinality
        if any(c is None for c in cards):
            raise ValueError("The transformer appears to have some continuous columns. Please provide only categorical or ordinal.")

        dimensionality = np.prod(cards)
        if self.verbose:
            print(f"Fitting with {dimensionality} dimensions")
            print(self._transformer.output_width)

        colnames = ["col" + str(i) for i in range(self._transformer.output_width)]

        if len(cards) != len(colnames):
            raise ValueError("Cardinality and column names must be the same length.")

        domain = Domain(colnames, cards)
        data = pd.DataFrame(train_data, columns=colnames)
        data = Dataset(df=data, domain=domain)
        workloads = get_rand_workloads(data, 100000, self.k)

        self.query_manager_torch = KWayMarginalQMTorch(data, 
                                                       workloads, 
                                                       verbose=True, 
                                                       device=self.device)
        
        true_answers_torch = self.query_manager_torch.get_answers(data)

        self.G = NeuralNetworkGenerator(self.query_manager_torch, 
                                        K=1000, 
                                        device=self.device, 
                                        init_seed=0,
                                        embedding_dim=512, 
                                        gen_dims=None,
                                        resample=False)
        
        self.algo = GEM(self.G, 
                        self.T, 
                        self.epsilon,
                        alpha=self.alpha, 
                        default_dir=None, 
                        verbose=True, 
                        seed=0,
                        loss_p=2, 
                        lr=self.lr, 
                        max_idxs=100, 
                        max_iters=100,
                        ema_weights=True, 
                        ema_weights_beta=self.ema_weights_beta)

        self.algo.fit(true_answers_torch)

    def sample(self, n):
        assert self.G is not None, "Please fit the synthesizer first."
        syndata = self.G.get_syndata(num_samples=n)
        df = self._unslide_range(syndata.df)
        return df