from dataclasses import dataclass
from typing import Optional, Union, Tuple
from pathlib import Path
import warnings

from sklearn.metrics import roc_auc_score
from ydnpd.pretraining.utils import load_data_for_classification, split_train_val
from ydnpd.pretraining.ft_transformer import FTTransformerModel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ModelConfig:
    """Base configuration for the transformer model"""
    dim: int = 32
    dim_out: int = 2
    depth: int = 6
    heads: int = 8
    attn_dropout: float = 0.1
    ff_dropout: float = 0.1
    batch_size: int = 128
    num_epochs: int = 20
    lr: float = 3e-4
    epsilon: Optional[float] = None  # Privacy budget for DP training


@dataclass
class PreTrainConfig:
    """Configuration for pretraining phase"""
    batch_size: int = 4
    num_epochs: int = 3
    lr: float = 3e-4


DataTuple = Tuple  # Type alias for the data tuple type


class TransformerTrainer:
    """Trainer class supporting three modes based on data availability:
    1. Public only (non-private training)
    2. Private only (DP training)
    3. Public + Private (pretrain on public, finetune on private with DP)
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        pretrain_config: Optional[PreTrainConfig] = None,
    ):
        """
        Initialize the trainer with specified configuration.

        Args:
            config: ModelConfig object containing model hyperparameters
            pretrain_config: Optional configuration for pretraining phase
        """
        self.config = config or ModelConfig()
        self.pretrain_config = pretrain_config or PreTrainConfig()
        self.model = None



    def _create_model(self, dp: bool = False, partial_dp: bool = False) -> FTTransformerModel:
        """Create a new FTTransformerModel instance with specified privacy settings"""
        model_params = {
            "dim": self.config.dim,
            "dim_out": self.config.dim_out,
            "depth": self.config.depth,
            "heads": self.config.heads,
            "attn_dropout": self.config.attn_dropout,
            "ff_dropout": self.config.ff_dropout,
            "batch_size": self.config.batch_size,
            "num_epochs": self.config.num_epochs,
            "lr": self.config.lr,
            "load_best_model_when_trained": True,
            "verbose": True,
        }

        if dp:
            if self.config.epsilon is None:
                raise ValueError("epsilon must be specified in config for private training")
            model_params.update({
                "dp": True,
                "epsilon": self.config.epsilon
            })

        if partial_dp:
            model_params.update({
                "partial_dp": True
            })

        return FTTransformerModel(**model_params)

    def execute(
        self,
        private_data: Optional[DataTuple] = None,
        public_data: Optional[DataTuple] = None,
        save_path: Optional[str] = None,
    ) -> dict:
        """
        Train and evaluate the model based on provided data.
        Mode is automatically determined by data availability:
        - public_data only -> non-private training
        - private_data only -> private training with DP
        - both -> pretrain on public_data, finetune on private_data with DP

        Args:
            private_data: Optional tuple containing private training data
            public_data: Optional tuple containing public training data
        """
        if private_data is None and public_data is None:
            raise ValueError("At least one of private_data or public_data must be provided")

        # Public data only - non-DP training
        if private_data is None:
            self.model = self._create_model(dp=False)

            X_cat_train_val, X_cont_train_val, _, _, y_train_val, _, cat_cardinalities, _ = public_data

            self.model.fit(
                X_cat_train_val,
                X_cont_train_val,
                y_train_val.flatten(),
                cat_cardinalities,
                X_cont_train_val.shape[1],
            )

            if save_path:
                self.model.save_torch(Path(save_path) / "model.pkl")

            return {"no-dp": self.evaluate(public_data)}

        # Private data only - DP training
        elif public_data is None:
            self.model = self._create_model(dp=True)

            X_cat_train_val, X_cont_train_val, _, _, y_train_val, _, cat_cardinalities, _ = private_data

            self.model.fit(
                X_cat_train_val,
                X_cont_train_val,
                y_train_val.flatten(),
                cat_cardinalities,
                X_cont_train_val.shape[1],
            )

            if save_path:
                self.model.save_torch(Path(save_path) / "model.pkl")

            return {"dp/private": self.evaluate(private_data)}

        else:

            results = {}

            # Both - pretrain on public, finetune on private with DP
            X_cat_train_val, X_cont_train_val, _, _, y_train_val, _, cat_cardinalities, _ = private_data
            X_cat_pre_train_val, X_cont_pre_train_val, _, _, y_pre_train_val, _, cat_cardinalities_pre, _ = public_data

            assert cat_cardinalities == cat_cardinalities_pre

            # Create model with partial DP and pretraining configuration
            self.model = self._create_model(dp=True, partial_dp=True)

            # Set up pretraining configuration
            pretrain_config = {
                'pre_epochs': self.pretrain_config.num_epochs,
                'pre_batch_size': self.pretrain_config.batch_size,
                'pre_lr': self.pretrain_config.lr,
            }
            self.model.partial_pretrain_config = pretrain_config

            self.model.fit_pre(
                X_cat_pre_train_val,
                X_cont_pre_train_val,
                y_pre_train_val.flatten(),
                cat_cardinalities_pre,
                X_cont_pre_train_val.shape[1])

            if save_path:
                self.model.save_torch(Path(save_path) / "model_pre.pkl")

            results |= {
                "pre/private": self.evaluate(private_data),
                "pre/public": self.evaluate(public_data)
                }

            # Fit model with pretraining and private finetuning
            self.model.fit(
                X_cat_train_val,
                X_cont_train_val,
                y_train_val.flatten(),
                cat_cardinalities,
                X_cont_train_val.shape[1]
            )

            if save_path:
                self.model.save_torch(Path(save_path) / "model.pkl")

            results |= {
                "dp/private": self.evaluate(private_data),
                "dp/public": self.evaluate(public_data)
                }

            return results

    def evaluate(self, data: DataTuple) -> dict[str, float]:
        """Evaluate model on test data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        X_cat_train_val, X_cont_train_val, X_cat_test, X_cont_test, y_train_val, y_test, _, _ = data
        (X_cat_train, X_cat_val, X_cont_train, X_cont_val,
         y_train, y_val) = split_train_val(X_cat_train_val, X_cont_train_val, y_train_val)

        results = {}
        for (name, (X_cat, X_cont, y)) in [("test", (X_cat_test, X_cont_test, y_test)),
                                           ("train", (X_cat_train, X_cont_train, y_train)),
                                           ("val", (X_cat_val, X_cont_val, y_val))]:

            y_pred = self.model.predict_proba(X_cat, X_cont)[:, 1]
            auc = roc_auc_score(y, y_pred)

            results[f"{name}/auc"] = auc

        return results

    @staticmethod
    def train_and_evaluate(
        private_data_pointer: Optional[Union[str, tuple]] = None,
        public_data_pointer: Optional[Union[str, tuple]] = None,
        config: Optional[ModelConfig] = None,
        pretrain_config: Optional[PreTrainConfig] = None,
        subsampling: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> dict[str, float]:
        """Convenience method to train and evaluate in one call"""
        trainer = TransformerTrainer(config, pretrain_config)

        private_data = None if private_data_pointer is None else load_data_for_classification(private_data_pointer, subsampling=subsampling)
        public_data = None if public_data_pointer is None else load_data_for_classification(public_data_pointer, subsampling=subsampling)

        return trainer.execute(private_data=private_data,
                               public_data=public_data,
                               save_path=save_path)
