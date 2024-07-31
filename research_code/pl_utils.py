from copy import deepcopy

import geotorch
import numpy as np
import pandas as pd
import torch
from lightning import LightningModule
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import r2_score
from torch import Tensor
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from green.spd_layers import BiMap
from green.spd_layers import LogMap
from green.spd_layers import Shrinkage
from green.wavelet_layers import PW_PLV
from green.wavelet_layers import CombinedPooling
from green.wavelet_layers import CrossCovariance
from green.wavelet_layers import CrossPW_PLV
from green.wavelet_layers import RealCovariance
from green.wavelet_layers import WaveletConv


def get_train_test_loaders(dataset,
                           train_indices,
                           test_indices,
                           batch_size=128,
                           num_workers=0,
                           shuffle=True,
                           pin_memory=True,
                           final_val=False
                           ):
    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=min(int(len(test_set) // batch_size), num_workers)
    )
    if final_val:
        test_set_final = deepcopy(test_set)
        test_set_final.dataset.padding = None
        test_set_final.dataset.n_epochs = 150
        test_loader_final = DataLoader(
            test_set_final,
            batch_size=1,
            num_workers=num_workers * 2
        )
        return train_loader, test_loader, test_loader_final

    return train_loader, test_loader


class GreenRegressorLM(LightningModule):
    def __init__(
            self,
            model,
            lr=1e-1,
            weight_decay=1e-5,
            lr_wavelet=None,
            data_type=torch.float32):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.predict_outputs = list()
        self.lr_wavelet = lr_wavelet
        self.data_type = data_type

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y_true = batch
        y_true = y_true.to(self.data_type)
        y_pred = self.model(x)
        loss = torch.nn.functional.mse_loss(y_pred.squeeze(-1), y_true)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.to(self.data_type)
        y_true = y_true.cpu()

        y_pred = self.model(x).cpu()
        test_loss = torch.nn.functional.mse_loss(y_pred.squeeze(-1),
                                                 y_true)
        test_score = r2_score(y_pred=y_pred.squeeze(-1).numpy(),
                              y_true=y_true.numpy())
        self.log("test_loss", test_loss)
        self.log("test_score", test_score)

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.to(self.data_type)
        y_true = y_true.cpu()

        y_pred = self.model(x).cpu()
        valid_loss = torch.nn.functional.mse_loss(y_pred.squeeze(-1),
                                                  y_true)
        valid_score = r2_score(y_pred=y_pred.squeeze(-1).numpy(),
                               y_true=y_true.numpy())
        self.log("valid_loss", valid_loss, prog_bar=True,)
        self.log("valid_score", valid_score, prog_bar=True,)

    def predict_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.to(self.data_type)
        y_true = y_true.cpu()

        y_pred = self.model(x).cpu()
        self.predict_outputs.append(pd.DataFrame(dict(
            y_pred=y_pred.squeeze(-1).numpy(),
            y_true=y_true.numpy().ravel(),
        )))
        return pd.DataFrame(dict(
            y_pred=y_pred.squeeze(-1).numpy(),
            y_true=y_true.numpy().ravel(),
        ))

    def configure_optimizers(self):
        params = list(self.named_parameters())
        if self.lr_wavelet is not None:
            def is_faster(n): return ('foi' in n) or ('fwhm' in n)
            grouped_parameters = [
                {"params": [p for n, p in params if not is_faster(n)],
                 'lr': self.lr},
                {"params": [p for n, p in params if is_faster(n)],
                 'lr': self.lr * self.lr_wavelet},
            ]
        else:
            grouped_parameters = self.parameters()

        optimizer = torch.optim.Adam(
            grouped_parameters,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer,
                                               mode='min',
                                               factor=.25),
                "monitor": "train_loss",
                "interval": "step",
                "frequency": 5,
            },
        }

    def on_predict_epoch_end(self, ):
        all_preds = pd.concat(self.predict_outputs)
        self.predict_outputs.clear()
        return all_preds


class GreenClassifierLM(LightningModule):
    def __init__(
            self,
            model,
            lr=1e-1,
            weight_decay=1e-5,
            lr_wavelet=None,
            data_type=torch.float32,
            use_age: bool = False,
            criterion: callable = torch.nn.functional.cross_entropy,
            scheduler_track: str = "train_loss"):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.predict_outputs = list()
        self.lr_wavelet = lr_wavelet
        self.data_type = data_type
        self.use_age = use_age
        self.criterion = criterion
        self.init_metrics = True
        self.scheduler_track = scheduler_track

    def training_step(self, batch, batch_idx):
        if self.current_epoch < 1:
            self.log("val_loss", torch.tensor(1e3))
            self.log("val_acc", torch.tensor(-1e3))
            self.init_metrics = False

        # training_step defines the train loop.
        # it is independent of forward
        if self.use_age:
            x, age, y_true = batch
            y_pred = self.model(x, age)
        else:
            x, y_true = batch
            y_pred = self.model(x)

        y_true = y_true.to(self.data_type)
        loss = self.criterion(y_pred, y_true)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        if self.use_age:
            x, age, y_true = batch
            y_pred = self.model(x, age)
        else:
            x, y_true = batch
            y_pred = self.model(x)

        test_loss = self.criterion(y_pred, y_true)

        y_pred = y_pred.cpu()
        y_true = y_true.to(self.data_type)
        y_true = y_true.cpu()

        test_score = balanced_accuracy_score(
            y_pred=torch.argmax(y_pred, dim=1).numpy(),
            y_true=torch.argmax(y_true, dim=1).numpy())
        self.log("test_loss", test_loss)
        self.log("test_score", test_score)

    def validation_step(self, batch, batch_idx):
        if self.use_age:
            x, age, y_true = batch
            y_pred = self.model(x, age)
        else:
            x, y_true = batch
            y_pred = self.model(x)

        val_loss = self.criterion(y_pred, y_true)
        y_pred = y_pred.cpu()
        y_true = y_true.to(self.data_type)
        y_true = y_true.cpu()

        val_acc = balanced_accuracy_score(
            y_pred=torch.argmax(y_pred, dim=1).numpy(),
            y_true=torch.argmax(y_true, dim=1).numpy())

        self.log("val_loss", val_loss, prog_bar=True,)
        self.log("val_acc", val_acc, prog_bar=True,)

    def predict_step(self, batch, batch_idx):
        if self.use_age:
            x, age, y_true = batch
            y_pred = self.model(x, age).cpu()
        else:
            x, y_true = batch
            y_pred = self.model(x).cpu()
        y_true = y_true.to(self.data_type)
        y_true = y_true.cpu()
        pred_acc = balanced_accuracy_score(
            y_pred=torch.argmax(y_pred, dim=1).numpy(),
            y_true=torch.argmax(y_true, dim=1).numpy())
        print("pred_acc = ", pred_acc)

        self.predict_outputs.append(pd.DataFrame(dict(
            y_pred=torch.argmax(y_pred, dim=1).numpy(),
            y_true=torch.argmax(y_true, dim=1).numpy(),
            y_pred_proba=tuple(y_pred.numpy()),
            y_true_proba=tuple(y_true.numpy())
        )))
        return pd.DataFrame(dict(
            y_pred=torch.argmax(y_pred, dim=1).numpy(),
            y_true=torch.argmax(y_true, dim=1).numpy(),
            y_pred_proba=tuple(y_pred.numpy()),
            y_true_proba=tuple(y_true.numpy())
        ))

    def configure_optimizers(self):
        params = list(self.named_parameters())
        if self.lr_wavelet is not None:
            def is_faster(n): return ('foi' in n) or ('fwhm' in n)
            grouped_parameters = [
                {"params": [p for n, p in params if not is_faster(n)],
                 'lr': self.lr},
                {"params": [p for n, p in params if is_faster(n)],
                 'lr': self.lr * self.lr_wavelet},
            ]
        else:
            grouped_parameters = self.parameters()

        optimizer = torch.optim.Adam(
            grouped_parameters,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer,
                                               mode='min',
                                               factor=.25,
                                               min_lr=1e-5,
                                               patience=5),
                "monitor": self.scheduler_track,
            },
        }

    def on_predict_epoch_end(self, ):
        all_preds = pd.concat(self.predict_outputs)
        self.predict_outputs.clear()
        return all_preds


def vectorize_upper(X: Tensor) -> Tensor:
    """Upper vectorisation of F SPD matrices with multiplication of
    off-diagonal terms by sqrt(2) to preserve norm.

    Parameters
    ----------
    X : Tensor
        (N) x F x C x C

    Returns
    -------
    Tensor
        (N) x (C (C + 1) / 2)
    """
    # Upper triangular
    d = X.shape[-1]
    triu_idx = torch.triu_indices(d, d, 1)
    if X.dim() == 4:  # batch
        X_out = torch.cat([
            torch.diagonal(X, dim1=-2, dim2=-1),
            X[:, :, triu_idx[0], triu_idx[1]] * np.sqrt(2)
        ], dim=-1)
        return X_out
    elif X.dim() == 3:  # single tensor
        return torch.cat([torch.diagonal(X, dim1=-2, dim2=-1),
                         X[triu_idx[0], triu_idx[1]] * np.sqrt(2)],
                         dim=-1)


def vectorize_upper_one(X: Tensor):
    """
    Upper vectorisation of a single SPD matrix with multiplication of
    off-diagonal terms by sqrt(2) to preserve norm.

    Parameters:
    -----------
    X : Tensor
        The covariance matrix of shape (N x P x P).

    Returns:
    --------
    X_vec : Tensor
        The vectorized covariance matrix of shape (N x P * (P + 1) / 2).
    """
    assert X.dim() == 3
    _, size, _ = X.shape
    triu_indices = torch.triu_indices(size, size, offset=1)
    if X.dim() == 3:  # batch of matrices
        X_vec = torch.cat([
            torch.diagonal(X, dim1=-2, dim2=-1),
            X[:, triu_indices[0], triu_indices[1]] * np.sqrt(2)
        ], dim=-1)
    elif X.dim() == 2:  # single matrix
        X_vec = torch.cat([
            torch.diagonal(X, dim1=-2, dim2=-1),
            X[triu_indices[0], triu_indices[1]] * np.sqrt(2)
        ], dim=-1)
    return X_vec


class Green(nn.Module):
    def __init__(self,
                 conv_layers: nn.Module,
                 pooling_layers: nn.Module,
                 spd_layers: nn.Module,
                 head: nn.Module,
                 proj: nn.Module,
                 use_age: bool = False
                 ):
        """
        Neural network model that processes EEG epochs using convolutional
        layers, follwed by the computation of SPD features.

        Parameters:
        -----------
        conv_layers : nn.Module
            The convolutional layers that operate on the raw EEG signals.
        pooling_layers : nn.Module
            The pooling layers that convert the the convolved signals
            to SPD (Symmetric Positive Definite) features.
        spd_layers : nn.Module
            The SPD layers that operate on the SPD features.
        head : nn.Module
            The head layer that acts in the Euclidean space.
        proj : nn.Module
            The projection layer that projects the SPD features to the
            Euclidean
            space.
        age : bool, optional
            Whether to include age in the model, by default False
        """
        super(Green, self).__init__()
        self.conv_layers = conv_layers
        self.pooling_layers = pooling_layers
        self.spd_layers = spd_layers
        self.proj = proj
        self.head = head
        self.use_age = use_age

    def forward(self, X: Tensor, age: Tensor = None):
        """
        Parameters
        ----------
        X : Tensor
            N x P x T
        age : _type_, optional
            N, by default None
        """
        X_hat = self.conv_layers(X)
        X_hat = self.pooling_layers(X_hat)
        X_hat = self.spd_layers(X_hat)
        X_hat = self.proj(X_hat)
        if isinstance(
            self.pooling_layers, RealCovariance
        ) or isinstance(self.pooling_layers, CombinedPooling):
            X_hat = vectorize_upper(X_hat)

        elif isinstance(
            self.pooling_layers, CrossCovariance
        ) or isinstance(
            self.pooling_layers, CrossPW_PLV
        ):
            X_hat = vectorize_upper_one(X_hat)

        X_hat = torch.flatten(X_hat, start_dim=1)
        if self.use_age:
            X_hat = torch.cat([X_hat, age.unsqueeze(-1)], dim=-1)
        X_hat = self.head(X_hat)
        return X_hat


def get_green(
    n_freqs: int = 15,
    kernel_width_s: int = 5,
    conv_stride: int = 5,
    oct_min: float = 0,
    oct_max: float = 5.5,
    random_f_init: bool = False,
    shrinkage_init: float = -3.,
    logref: str = 'logeuclid',
    dropout: float = .333,
    n_ch: int = 21,
    hidden_dim: int = 32,
    sfreq: int = 125,
    dtype: torch.dtype = torch.float32,
    pool_layer: nn.Module = RealCovariance(),
    bi_out: int = None,
    out_dim: int = 1,
    use_age: bool = False,
    orth_weights=True
):
    """
    Helper function to get a Green model.

    Parameters
    ----------
    n_freqs : int, optional
        Number of main frequencies in the wavelet family, by default 15
    kernel_width_s : int, optional
        Width of the kernel in seconds for the wavelets, by default 5
    conv_stride : int, optional
        Stride of the convolution operation for the wavelets, by default 5
    oct_min : float, optional
        Minimum foi in octave, by default 0
    oct_max : float, optional
        Maximum foi in octave, by default 5.5
    random_f_init : bool, optional
        Whether to randomly initialize the foi, by default False
    shrinkage_init : float, optional
        Initial shrinkage value before applying sigmoid funcion, by default -3.
    logref : str, optional
        Reference matrix used for LogEig layer, by default 'logeuclid'
    dropout : float, optional
        Dropout rate for FC layers, by default .333
    n_ch : int, optional
        Number of channels, by default 21
    hidden_dim : int, optional
        Dimension of the hidden layer, if None no hidden layer, by default 32
    sfreq : int, optional
        Sampling frequency, by default 125
    dtype : torch.dtype, optional
        Data type of the tensors, by default torch.float32
    pool_layer : nn.Module, optional
        Pooling layer, by default RealCovariance()
    bi_out : int, optional
        Dimension of the output layer after BiMap, by default None
    out_dim : int, optional
        Dimension of the output layer, by default 1
    use_age : bool, optional
        Whether to include age in the model, by default False

    Returns
    -------
    Green
        The Green model
    """

    # Convolution
    cplx_dtype = torch.complex128 if (
        dtype == torch.float64) else torch.complex64
    if random_f_init:
        foi_init = np.random.uniform(oct_min, oct_max, size=n_freqs)
        fwhm_init = -np.random.uniform(oct_min - 1, oct_max - 1, size=n_freqs)
    else:
        foi_init = np.linspace(oct_min, oct_max, n_freqs)
        fwhm_init = -np.linspace(oct_min - 1, oct_max - 1, n_freqs)

    conv_layers = nn.Sequential(*[
        WaveletConv(
            kernel_width_s=kernel_width_s,
            sfreq=sfreq,
            foi_init=foi_init,
            fwhm_init=fwhm_init,
            stride=conv_stride,
            dtype=cplx_dtype,
            scaling='oct'
        )])

    if isinstance(
            pool_layer, RealCovariance
    ) or isinstance(
            pool_layer, PW_PLV):
        n_compo = n_ch
        feat_dim = int(n_freqs * n_compo * (n_compo + 1) / 2)

    elif isinstance(pool_layer, CrossCovariance):
        n_compo = int(n_ch * n_freqs)
        feat_dim = int(n_compo * (n_compo + 1) / 2)
        n_freqs = None

    elif isinstance(pool_layer, CombinedPooling):
        pool_layer_0 = pool_layer.pooling_layers[0]
        if isinstance(
            pool_layer_0, RealCovariance
        ) or isinstance(
                pool_layer_0, PW_PLV):
            n_compo = n_ch
            feat_dim = int(n_freqs * n_compo * (n_compo + 1) /
                           2) * len(pool_layer.pooling_layers)
            n_freqs = n_freqs * len(pool_layer.pooling_layers)

        elif isinstance(pool_layer_0, CrossCovariance
                        ) or isinstance(pool_layer_0, CrossPW_PLV):
            n_compo = int(n_ch * n_freqs)
            feat_dim = int(n_compo * (n_compo + 1) / 2) * \
                len(pool_layer.pooling_layers)
            n_freqs = len(pool_layer.pooling_layers)

    # pooling
    pool_layer = pool_layer

    # SPD layers
    if shrinkage_init is None:
        spd_layers_list = [nn.Identity()]
    else:
        spd_layers_list = [Shrinkage(n_freqs=n_freqs,
                                     size=n_compo,
                                     init_shrinkage=shrinkage_init,
                                     learnable=True
                                     )]
    if bi_out is not None:
        for bo in bi_out:
            bimap = BiMap(d_in=n_compo,
                          d_out=bo,
                          n_freqs=n_freqs)
            if orth_weights:
                geotorch.orthogonal(bimap, 'weight')
            spd_layers_list.append(bimap)

            n_compo = bo

        if n_freqs is None:
            feat_dim = int(n_compo * (n_compo + 1) / 2)
        else:
            feat_dim = int(n_freqs * n_compo * (n_compo + 1) / 2)

    if use_age:
        feat_dim += 1
    spd_layers = nn.Sequential(*spd_layers_list)

    # Projection to tangent space
    proj = LogMap(size=n_compo,
                  n_freqs=n_freqs,
                  ref=logref,
                  momentum=0.9,
                  reg=1e-4)

    # Head
    if hidden_dim is None:
        head = torch.nn.Sequential(*[
            torch.nn.BatchNorm1d(feat_dim,
                                 dtype=dtype),
            torch.nn.Dropout(
                p=dropout) if dropout is not None else nn.Identity(),
            torch.nn.Linear(feat_dim,
                            out_dim,
                            dtype=dtype),
        ])
    else:
        # add multiple FC layers
        sequential_list = []
        for hd in hidden_dim:
            sequential_list.extend([
                torch.nn.BatchNorm1d(feat_dim,
                                     dtype=dtype),
                torch.nn.Dropout(
                    p=dropout) if dropout is not None else nn.Identity(),
                torch.nn.Linear(feat_dim,
                                hd,
                                dtype=dtype),
                torch.nn.GELU()
            ])
            feat_dim = hd
        sequential_list.extend([
            torch.nn.BatchNorm1d(feat_dim,
                                 dtype=dtype),
            torch.nn.Dropout(
                p=dropout) if dropout is not None else nn.Identity(),
            torch.nn.Linear(feat_dim,
                            out_dim,
                            dtype=dtype)
        ])
        head = torch.nn.Sequential(*sequential_list)

    # Gather everything
    model = Green(
        conv_layers=conv_layers,
        pooling_layers=pool_layer,
        spd_layers=spd_layers,
        head=head,
        proj=proj,
        use_age=use_age
    )
    return model


class GreenG2(nn.Module):
    def __init__(self,
                 spd_layers: nn.Module,
                 head: nn.Module,
                 proj: nn.Module,
                 ):
        """
        Version of the Green that uses pre-computed covariance matrices.


        Parameters:
        -----------
        spd_layers : nn.Module
            The SPD layers that operate on the SPD features.
        head : nn.Module
            The head layer that acts in the Euclidean space.
        proj : nn.Module
            The projection layer that projects the SPD features to the
            Euclidean
            space.
        """
        super(GreenG2, self).__init__()
        self.spd_layers = spd_layers
        self.proj = proj
        self.head = head

    def forward(self, X: Tensor, ):
        """
        Parameters
        ----------
        X : Tensor
            N x P x T
        """
        X_hat = self.spd_layers(X)
        X_hat = self.proj(X_hat)
        X_hat = vectorize_upper(X_hat)
        X_hat = torch.flatten(X_hat, start_dim=1)
        X_hat = self.head(X_hat)
        return X_hat


def get_green_g2(
    n_freqs: int = 15,
    shrinkage_init: float = -3.,
    logref: str = 'logeuclid',
    dropout: float = .333,
    n_ch: int = 21,
    hidden_dim: int = 32,
    dtype: torch.dtype = torch.float32,
    bi_out: int = None,
    out_dim: int = 1,
    use_age: bool = False,
    orth_weights=True
):
    """
    Helper function to get a Green model.

    Parameters
    ----------
    shrinkage_init : float, optional
        Initial shrinkage value before applying sigmoid funcion, by default -3.
    logref : str, optional
        Reference matrix used for LogEig layer, by default 'logeuclid'
    dropout : float, optional
        Dropout rate for FC layers, by default .333
    n_ch : int, optional
        Number of channels, by default 21
    hidden_dim : int, optional
        Dimension of the hidden layer, if None no hidden layer, by default 32
    dtype : torch.dtype, optional
        Data type of the tensors, by default torch.float32
    bi_out : int, optional
        Dimension of the output layer after BiMap, by default None
    out_dim : int, optional
        Dimension of the output layer, by default 1
    use_age : bool, optional
        Whether to include age in the model, by default False

    Returns
    -------
    Green
        The Green model
    """
    n_compo = n_ch
    # SPD layers
    if shrinkage_init is None:
        spd_layers_list = [nn.Identity()]
    else:
        spd_layers_list = [Shrinkage(n_freqs=n_freqs,
                                     size=n_compo,
                                     init_shrinkage=shrinkage_init,
                                     learnable=True
                                     )]
    if bi_out is not None:
        for bo in bi_out:
            bimap = BiMap(d_in=n_compo,
                          d_out=bo,
                          n_freqs=n_freqs)
            if orth_weights:
                geotorch.orthogonal(bimap, 'weight')
            spd_layers_list.append(bimap)

            n_compo = bo

        if n_freqs is None:
            feat_dim = int(n_compo * (n_compo + 1) / 2)
        else:
            feat_dim = int(n_freqs * n_compo * (n_compo + 1) / 2)

    if use_age:
        feat_dim += 1
    spd_layers = nn.Sequential(*spd_layers_list)

    # Projection to tangent space
    proj = LogMap(size=n_compo,
                  n_freqs=n_freqs,
                  ref=logref,
                  momentum=0.9,
                  reg=1e-4)

    # Head
    if hidden_dim is None:
        head = torch.nn.Sequential(*[
            torch.nn.BatchNorm1d(feat_dim,
                                 dtype=dtype),
            torch.nn.Dropout(
                p=dropout) if dropout is not None else nn.Identity(),
            torch.nn.Linear(feat_dim,
                            out_dim,
                            dtype=dtype),
        ])
    else:
        # add multiple FC layers
        sequential_list = []
        for hd in hidden_dim:
            sequential_list.extend([
                torch.nn.BatchNorm1d(feat_dim,
                                     dtype=dtype),
                torch.nn.Dropout(
                    p=dropout) if dropout is not None else nn.Identity(),
                torch.nn.Linear(feat_dim,
                                hd,
                                dtype=dtype),
                torch.nn.GELU()
            ])
            feat_dim = hd
        sequential_list.extend([
            torch.nn.BatchNorm1d(feat_dim,
                                 dtype=dtype),
            torch.nn.Dropout(
                p=dropout) if dropout is not None else nn.Identity(),
            torch.nn.Linear(feat_dim,
                            out_dim,
                            dtype=dtype)
        ])
        head = torch.nn.Sequential(*sequential_list)

    # Gather everything
    model = GreenG2(
        spd_layers=spd_layers,
        head=head,
        proj=proj,
    )
    return model
