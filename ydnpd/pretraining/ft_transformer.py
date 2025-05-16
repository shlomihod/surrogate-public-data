
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch import einsum
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm
from einops import rearrange
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from ydnpd.pretraining.consts import VAL_PROP, RANDOM_STATE


## NOTE: Below is modified version of https://github.com/lucidrains/tab-transformer-pytorch
# tab_transformer_pytorch implementation of FTTransformer
# to make it compatible with opacus (by changing certain modules).
# Thanks to the authors of that repo!
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("input to attention contains nans or infs")

        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        mult=4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * mult),
                    nn.ReLU(),
                    nn.Dropout(ff_dropout),
                    nn.Linear(dim * mult, dim)
                )
            ]))

    def forward(self, x, return_attn=False):
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("input to transformer contains nans or infs")

        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.dim = dim
        self.num_numerical_types = num_numerical_types

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("input to NumericalEmbedder contains nans or infs")

        weights = torch.randn(self.num_numerical_types, self.dim, device=x.device, requires_grad=True)
        biases = torch.randn(self.num_numerical_types, self.dim, device=x.device, requires_grad=True)
        x = rearrange(x, 'b n -> b n 1')
        return x * weights + biases

class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head=16,
        dim_out=1,
        num_special_tokens=2,
        attn_dropout=0.,
        ff_dropout=0.
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        self.dim = dim

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, return_attn=False):
        if torch.isnan(x_categ).any() or torch.isinf(x_categ).any():
            raise ValueError("cat input contains nans or infs")

        if torch.isnan(x_numer).any() or torch.isinf(x_numer).any():
            raise ValueError("num input contains nans or infs")

        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        xs = []
        if self.num_unique_categories > 0:

            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        x = torch.cat(xs, dim=1)

        b = x.shape[0]
        cls_tokens = torch.randn(b, 1, self.dim, device=x.device, requires_grad=True)
        x = torch.cat((cls_tokens, x), dim=1)

        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("input to transformer contains nans or infs")

        x, attns = self.transformer(x, return_attn=True)

        x = x[:, 0]

        logits = self.to_logits(x)

        if not return_attn:
            return logits

        return logits, attns

class FTTransformerModel(BaseEstimator):
    def __init__(self,
                 dim = 32,
                 dim_out = 2,
                 depth = 6,
                 heads = 8,
                 attn_dropout = 0.1,
                 ff_dropout = 0.1,
                 batch_size=1,
                 num_epochs=50,
                 lr=3e-4,
                 load_best_model_when_trained=True,
                 verbose=False,
                 dp=False,
                 delta=1e-5,
                 max_grad_norm=1.0,
                 minority_weight=1.0,
                 epsilon=None,
                 patience=10,
                 partial_dp=False,
                 partial_pretrain_config=None):
        super(FTTransformerModel, self).__init__()

        self.verbose = verbose
        self.dim = dim
        self.dim_out = dim_out
        self.depth = depth
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.minority_weight = minority_weight
        self.epsilon = epsilon
        self.patience = patience

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.load_best_model_when_trained = load_best_model_when_trained
        self.best_model_dict = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device='cpu'
        self.dp = dp
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.model = None
        self.partial_dp = partial_dp
        self.partial_pretrain_config = partial_pretrain_config

    def build_model(self, categories, num_continuous, dim, dim_out, depth, heads, attn_dropout, ff_dropout):
        model = FTTransformer(
            categories = categories,
            num_continuous = num_continuous,
            dim = dim,
            dim_out = dim_out,
            depth = depth,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)

        return model

    def cross_entropy_loss(self, pred, y_batch, class_weights=None):
        if class_weights is not None:
            weights = class_weights.clone().detach().to(torch.float32).to(self.device)
            return F.cross_entropy(pred, y_batch, weight=weights)
        return F.cross_entropy(pred, y_batch)

    def _pretrain(self, X_cat_pre, X_cont_pre, y_pre, categories, num_continuous, pre_epochs, pre_batch_size, pre_lr, focal_class_weights=None):

        # do a pretraining step without dp if partial_dp is true
        # i.e. train the model on a given pretraining dataset before the private training
        if self.model is None:
            self.model = self.build_model(
                categories=categories,
                num_continuous=num_continuous,
                dim=self.dim,
                dim_out=self.dim_out,
                depth=self.depth,
                heads=self.heads,
                attn_dropout=self.attn_dropout,
                ff_dropout=self.ff_dropout,
            ).to(self.device)

        self.model.train()

        # move data to device
        X_cat_pre, X_cont_pre, y_pre = X_cat_pre.to(self.device), X_cont_pre.to(self.device), y_pre.to(self.device)

        # TODO: Fix the situation where there are no continous/categorical
        # if len(X_cat_pre) == 0:
        #     X_cat_train, X_cat_val = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        #     X_cont_train, X_cont_val, y_train, y_val = train_test_split(X_cont_pre,
        #                                                                 y_pre,
        #                                                                 test_size=0.1,
        #                                                                 random_state=42)
        # elif len(X_cont_pre) == 0:
        #     X_cat_train, X_cat_val, y_train, y_val = train_test_split(X_cat_pre,
        #                                                               y_pre,
        #                                                               test_size=0.1,
        #                                                               random_state=42)
        #     X_cont_train, X_cont_val = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        # else:

        X_cat_train, X_cat_val, X_cont_train, X_cont_val, y_train, y_val = train_test_split(
            X_cat_pre,
            X_cont_pre,
            y_pre,
            test_size=VAL_PROP,
            random_state=RANDOM_STATE
        )

        if len(X_cat_train) > 0 and len(X_cont_train) > 0:
            train_dataset = TensorDataset(X_cat_train, X_cont_train, y_train)
        elif len(X_cat_train) > 0:
            train_dataset = TensorDataset(X_cat_train, y_train)
        elif len(X_cont_train) > 0:
            train_dataset = TensorDataset(X_cont_train, y_train)
        else:
            raise ValueError("both categorical and continuous pretraining data cannot be empty")

        train_loader = DataLoader(train_dataset, batch_size=pre_batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=pre_lr)

        best_loss = np.inf
        epochs_without_improvement = 0

        for epoch in range(pre_epochs):
            self.model.train()
            batch_progress = tqdm(train_loader, desc=f'pretraining epoch {epoch+1}/{pre_epochs}', unit='batch') if self.verbose else train_loader
            for batch in batch_progress:
                if len(batch) == 3:
                    X_batch_cat, X_batch_cont, y_batch = batch
                    X_batch_cat, X_batch_cont, y_batch = X_batch_cat.to(self.device), X_batch_cont.to(self.device), y_batch.to(self.device)
                elif len(batch) == 2:
                    if len(X_cat_train) > 0:
                        X_batch_cat, y_batch = batch
                        X_batch_cat, y_batch = X_batch_cat.to(self.device), y_batch.to(self.device)
                        X_batch_cont = torch.tensor([]).to(self.device)
                    else:
                        X_batch_cont, y_batch = batch
                        X_batch_cont, y_batch = X_batch_cont.to(self.device), y_batch.to(self.device)
                        X_batch_cat = torch.tensor([]).to(self.device)

                optimizer.zero_grad()
                output = self.model(X_batch_cat, X_batch_cont)
                y_batch = y_batch.type(torch.LongTensor).to(self.device)

                loss = self.cross_entropy_loss(output, y_batch)
                loss.backward()

                optimizer.step()
                if self.verbose:
                    batch_progress.set_description(f'pretraining Epoch {epoch+1}/{pre_epochs} loss: {loss.item():.4f}')
                    batch_progress.refresh()

            self.model.eval()
            with torch.no_grad():
                if len(X_cat_val) > 0 and len(X_cont_val) > 0:
                    val_output = self.model(X_cat_val, X_cont_val)
                elif len(X_cat_val) > 0:
                    val_output = self.model(X_cat_val, torch.tensor([]).to(self.device))
                elif len(X_cont_val) > 0:
                    val_output = self.model(torch.tensor([]).to(self.device), X_cont_val)
                else:
                    raise ValueError("Both categorical and continuous validation data cannot be empty")

                y_val = y_val.type(torch.LongTensor).to(self.device)
                val_loss = self.cross_entropy_loss(val_output, y_val)

            if val_loss < best_loss:
                best_loss = val_loss
                self.best_model_dict = self.model.state_dict()
                epochs_without_improvement = 0
                if self.verbose:
                    print(f"pretraining val loss - new best: {best_loss}")
            else:
                epochs_without_improvement += 1

            if self.verbose:
                print(f"pretraining epoch {epoch+1}, validation Loss: {val_loss.item()}, "
                      f"epochs without improvement: {epochs_without_improvement}/{self.patience}")

            if epochs_without_improvement >= self.patience:
                print(f"stopping pretraining early at epoch {epoch+1} - no improvement in validation loss for {self.patience} consecutive epochs.")
                break

        # Load the best model state found during pretraining
        if self.load_best_model_when_trained and self.best_model_dict is not None:
            self.model.load_state_dict(self.best_model_dict)

    def fit_pre(self,
                pre_X_cat,
                pre_X_cont,
                pre_y,
                categories,
                num_continuous,
                focal_class_weights=None):

        print(f"{categories=}")

        # if partial_dp is true, do a pretraining step without dp first
        if self.partial_dp:  # and self.partial_pretrain_config is not None:
            if self.partial_pretrain_config is None:
                raise ValueError("`partial_pretrain_config` should be set")

            self.dp = True
            pre_epochs = self.partial_pretrain_config.get('pre_epochs', 10)
            pre_batch_size = self.partial_pretrain_config.get('pre_batch_size', 64)
            pre_lr = self.partial_pretrain_config.get('pre_lr', 3e-4)

            self._pretrain(
                X_cat_pre=pre_X_cat,
                X_cont_pre=pre_X_cont,
                y_pre=pre_y,
                categories=categories,
                num_continuous=num_continuous,
                pre_epochs=pre_epochs,
                pre_batch_size=pre_batch_size,
                pre_lr=pre_lr,
                focal_class_weights=focal_class_weights
            )

    def fit(self,
            X_cat_train,
            X_cont_train,
            y_train,
            categories,
            num_continuous,
            ):

            # use_class_weights=False,
            # focal_class_weights=None):

        # # if partial_dp is true, do a pretraining step without dp first
        # if self.partial_dp and self.partial_pretrain_config is not None:
        #     self.dp = True
        #     pre_X_cat = self.partial_pretrain_config.get('X_cat_pre', torch.tensor([]))
        #     pre_X_cont = self.partial_pretrain_config.get('X_cont_pre', torch.tensor([]))
        #     pre_y = self.partial_pretrain_config.get('y_pre', torch.tensor([]))
        #     pre_epochs = self.partial_pretrain_config.get('pre_epochs', 10)
        #     pre_batch_size = self.partial_pretrain_config.get('pre_batch_size', 64)
        #     pre_lr = self.partial_pretrain_config.get('pre_lr', 3e-4)

        #     self._pretrain(
        #         X_cat_pre=pre_X_cat,
        #         X_cont_pre=pre_X_cont,
        #         y_pre=pre_y,
        #         categories=categories,
        #         num_continuous=num_continuous,
        #         pre_epochs=pre_epochs,
        #         pre_batch_size=pre_batch_size,
        #         pre_lr=pre_lr,
        #         focal_class_weights=focal_class_weights
        #     )

        if self.partial_dp and self.model is None:
            raise ValueError("`fit_pre` should be called first")

        # after pretraining, proceed with dp training, or non-dp if dp=False
        if self.model is None:
            self.model = self.build_model(
                categories=categories,
                num_continuous=num_continuous,
                dim=self.dim,
                dim_out=self.dim_out,
                depth=self.depth,
                heads=self.heads,
                attn_dropout=self.attn_dropout,
                ff_dropout=self.ff_dropout,
            ).to(self.device)

        self.model.train()

        num_classes = self.dim_out

        y_min = y_train.min().item()
        y_max = y_train.max().item()

        if y_min < 0 or y_max >= num_classes:
            raise ValueError(f"target labels should be in the range [0, {num_classes - 1}] - found labels in [{y_min}, {y_max}]")

        for i, cat_size in enumerate(categories):
            max_val = X_cat_train[:, i].max().item()
            if max_val >= cat_size:
                raise ValueError(f"training data for feature {i} has a category index {max_val} >= category size {cat_size}")

        class_weights = None

        X_cat_train, X_cont_train, y_train = X_cat_train.to(self.device), X_cont_train.to(self.device), y_train.to(self.device)

        if len(X_cat_train) == 0:
            X_cat_train, X_cat_val = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
            X_cont_train, X_cont_val, y_train, y_val = train_test_split(X_cont_train,
                                                                        y_train,
                                                                        test_size=VAL_PROP,
                                                                        random_state=RANDOM_STATE)
        elif len(X_cont_train) == 0:
            X_cat_train, X_cat_val, y_train, y_val = train_test_split(X_cat_train,
                                                                      y_train,
                                                                      test_size=VAL_PROP,
                                                                      random_state=RANDOM_STATE)
            X_cont_train, X_cont_val = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        else:
            X_cat_train, X_cat_val, X_cont_train, X_cont_val, y_train, y_val = train_test_split(X_cat_train,
                                                                                                X_cont_train,
                                                                                                y_train,
                                                                                                test_size=VAL_PROP,
                                                                                                random_state=RANDOM_STATE)

        if len(X_cat_train) > 0 and len(X_cont_train) > 0:
            train_dataset = TensorDataset(X_cat_train, X_cont_train, y_train)
        elif len(X_cat_train) > 0:
            train_dataset = TensorDataset(X_cat_train, y_train)
        elif len(X_cont_train) > 0:
            train_dataset = TensorDataset(X_cont_train, y_train)
        else:
            raise ValueError("both categorical and continuous training data cannot be empty")

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.dp:
            privacy_engine = PrivacyEngine()
            self.model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                epochs=self.num_epochs,
                max_grad_norm=self.max_grad_norm,
                grad_sample_mode="functorch"
            )

        best_accuracy = 0.0
        best_auc = 0.0
        best_loss = np.inf
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            batch_progress = tqdm(train_loader, desc=f'epoch {epoch+1}/{self.num_epochs}', unit='batch') if self.verbose else train_loader
            for batch in batch_progress:
                if len(batch) == 3:
                    X_batch_cat, X_batch_cont, y_batch = batch
                    X_batch_cat, X_batch_cont, y_batch = X_batch_cat.to(self.device), X_batch_cont.to(self.device), y_batch.to(self.device)
                elif len(batch) == 2:
                    if len(X_cat_train) > 0:
                        X_batch_cat, y_batch = batch
                        X_batch_cat, y_batch = X_batch_cat.to(self.device), y_batch.to(self.device)
                        X_batch_cont = torch.tensor([]).to(self.device)
                    else:
                        X_batch_cont, y_batch = batch
                        X_batch_cont, y_batch = X_batch_cont.to(self.device), y_batch.to(self.device)
                        X_batch_cat = torch.tensor([]).to(self.device)

                optimizer.zero_grad()

                output = self.model(X_batch_cat, X_batch_cont)
                y_batch = y_batch.type(torch.LongTensor).to(self.device)

                loss = self.cross_entropy_loss(output, y_batch, class_weights)
                loss.backward()

                optimizer.step()

                if self.verbose:
                    batch_progress.set_description(f'epoch {epoch+1}/{self.num_epochs} loss: {loss.item():.4f}')
                    batch_progress.refresh()

            self.model.eval()
            with torch.no_grad():
                if len(X_cat_val) > 0 and len(X_cont_val) > 0:
                    val_output = self.model(X_cat_val, X_cont_val)
                elif len(X_cat_val) > 0:
                    val_output = self.model(X_cat_val, torch.tensor([]).to(self.device))
                elif len(X_cont_val) > 0:
                    val_output = self.model(torch.tensor([]).to(self.device), X_cont_val)
                else:
                    raise ValueError("both categorical and continuous validation data cannot be empty")

                y_val = y_val.type(torch.LongTensor).to(self.device)
                val_loss = self.cross_entropy_loss(val_output, y_val)

                predict_proba = val_output.softmax(dim=-1)
                val_preds_binary = (predict_proba[:, 1] > 0.5).cpu().numpy().astype(int)
                val_accuracy = accuracy_score(y_val.cpu().numpy(), val_preds_binary)
                val_auc = roc_auc_score(y_val.cpu().numpy(), predict_proba.cpu().numpy()[:, 1])

            if val_loss < best_loss:
                best_loss = val_loss
                best_accuracy = val_accuracy
                best_auc = val_auc
                self.best_model_dict = self.model.state_dict()
                if self.verbose:
                    print(f"val loss - new best: {best_loss}")
                    print(f"val accuracy - new best: {best_accuracy}")
                    print(f"val AUC - new best: {best_auc}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.verbose:
                print(f"epoch {epoch+1}, validation loss: {val_loss.item()}, epochs without improvement: {epochs_without_improvement}/{self.patience}")

            if epochs_without_improvement >= self.patience:
                print(f"stopping early at epoch {epoch+1}. no improvement in validation loss for {self.patience} consecutive epochs.")
                print(f"epoch {epoch+1}, validation loss: {val_loss.item()}, epochs without improvement: {epochs_without_improvement}/{self.patience}")
                print(f"val accuracy - new best: {best_accuracy}")
                print(f"val auc - new best: {best_auc}")
                break

            if epoch % 4 == 0:
                print(f"reached {epoch+1}.")
                print(f"epoch {epoch+1}, val loss: {val_loss.item()}, epochs without improvement: {epochs_without_improvement}/{self.patience}")
                print(f"val accuracy - new best: {best_accuracy}")
                print(f"val auc - new best: {best_auc}")

        if self.load_best_model_when_trained and self.best_model_dict is not None:
            self.model.load_state_dict(self.best_model_dict)

    def load_best_model(self):
        if self.best_model_dict is not None:
            self.model.load_state_dict(self.best_model_dict)
        else:
            print("no best model saved - run the training first")

    def predict(self, X_cat_test, X_cont_test, binary=False, threshold=0.5):
        with torch.no_grad():
            if torch.isnan(X_cat_test).any() or torch.isinf(X_cat_test).any():
                raise ValueError("nans or infs found in X_cat_test")

            if torch.isnan(X_cont_test).any() or torch.isinf(X_cont_test).any():
                raise ValueError("nans or infs found in X_cont_test")

            X_cat_test, X_cont_test = X_cat_test.to(self.device), X_cont_test.to(self.device)
            predictions = self.model(X_cat_test, X_cont_test)
            proba = predictions.softmax(dim=-1)
            if binary:
                predictions = (proba[:, 1] > threshold).cpu().numpy().astype(int)
            else:
                predictions = predictions.cpu().numpy()
            return predictions

    def predict_proba(self, X_cat_test, X_cont_test):
        with torch.no_grad():
            if torch.isnan(X_cat_test).any() or torch.isinf(X_cat_test).any():
                raise ValueError("nans or infs found in X_cat_test")

            if torch.isnan(X_cont_test).any() or torch.isinf(X_cont_test).any():
                raise ValueError("nans or infs found in X_cont_test")

            X_cat_test, X_cont_test = X_cat_test.to(self.device), X_cont_test.to(self.device)
            predictions = self.model(X_cat_test, X_cont_test)
            proba = predictions.softmax(dim=-1)
            return proba.cpu().numpy()

    def save_torch(self, path: str):
        print(f"save model to `{path}`")
        torch.save(self.model.state_dict(), path)
