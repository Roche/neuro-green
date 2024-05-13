
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


def _compute_gaborwavelet(
        tt: nn.Parameter,
        foi: nn.Parameter,
        fwhm: nn.Parameter,
        dtype: torch.dtype = torch.complex64,
        sfreq: int = 250,
        scaling='oct',
        min_foi_oct=-2,
        max_foi_oct=6,
        min_fwhm_oct=-6,
        max_fwhm_oct=1
):
    """
    Compute the Gabor wavelet filterbank for a given set of frequencies and
    full-width at half-maximums.

    Parameters
    ----------
    tt : torch.Tensor
        The time vector at expected sampling frequency.
    foi : torch.Tensor
        The center frequencies of the wavelets in octaves.
    fwhm : torch.Tensor
        The full-width at half-maximums of the wavelets in octaves. This
        parameter is a time domain parameter. It matches the formalism
        described in "A betterway to define and describe Morlet wavelets for
        time-frequency analysis"by Michael X Cohen. It is related to the time
        domain standard deviationby fwhm =  std / 2 * sqrt(2 * log(2))
    dtype : torch.dtype, optional
        The dtype of the output wavelets, by default torch.complex64
    sfreq : int, optional
        The sampling frequency of the data, by default 250
    scaling : str, optional
        The scaling of the wavelets, if 'oct' the wavelets are scaled using
        the octave scaling described in meeglette, by default 'oct'
    min_foi_oct : int, optional
        The minimum center frequency in octaves used to ensure gradient flow,
        by default -2
    max_foi_oct : int, optional
        by default 6
    min_fwhm_oct : int, optional
        by default -6
    max_fwhm_oct : int, optional
        by default 1
    """

    foi_oct = 2**torch.clamp(foi, min_foi_oct, max_foi_oct)
    fwhm_oct = 2**torch.clamp(fwhm, min_fwhm_oct, max_fwhm_oct)

    wavelets = torch.stack([
        torch.exp(2j * np.pi * f * tt) * torch.exp(
            -4 * np.log(2) * tt**2 / h**2
        ) for f, h in zip(foi_oct, fwhm_oct)
    ], dim=0).to(dtype)

    wav_norm = wavelets / torch.linalg.norm(wavelets, dim=-1, keepdim=True)
    if scaling == 'oct':
        wav_norm *= np.sqrt(2.0 / sfreq) * \
            torch.sqrt(np.log(2) * foi_oct).unsqueeze(1)
    return wav_norm


class WaveletConv(nn.Module):

    def __init__(self,
                 kernel_width_s: float,
                 sfreq: float = None,
                 foi_init: np.ndarray = None,
                 fwhm_init: np.ndarray = None,
                 padding: str = 0,
                 dtype: torch.dtype = torch.complex64,
                 stride: int = 1,
                 scaling: str = 'oct'):
        """Parametrized complex wavelet convolution layer.


        Parameters
        ----------
        kernel_width_s : float
            The width of the wavelet kernel in seconds.
        sfreq : float, optional
            The sampling frequency of the data, by default None
        foi_init : np.ndarray, optional
            The initial center frequencies of the wavelets in octaves,
            by default None
        fwhm_init : np.ndarray, optional
            The initial full-width at half-maximums of the wavelets in
            octaves, by default None
        padding : str, optional
            Padding mode for the convolution, by default 0
        dtype : torch.dtype, optional
            The data type of the wavelets, by default torch.complex64
        stride : int, optional
           The stride of the convolution, by default 1
        scaling : str, optional
            The scaling of the wavelets, if 'oct' the wavelets are scaled
            using the octave scaling described in meeglette, by default 'oct'

        """

        super(WaveletConv, self).__init__()

        tmax = kernel_width_s / 2
        tmin = -tmax
        self.tt = nn.Parameter(torch.linspace(tmin, tmax, int(
            kernel_width_s * sfreq)), requires_grad=False)
        self.n_wavelets = len(foi_init)
        self.sfreq = sfreq
        self.kernel_width_s = kernel_width_s
        self.tmax = tmax
        self.tmin = tmin
        self.dtype = dtype
        self.padding = padding
        self.stride = stride
        self.scaling = scaling

        self.foi = nn.Parameter(torch.Tensor(foi_init), requires_grad=True)
        self.fwhm = nn.Parameter(torch.Tensor(fwhm_init), requires_grad=True)

    def forward(self, X: Tensor):
        """
        Forward pass of the complex wavelet module.

        Parameters:
        -----------
        X : Tensor
            Input data of shape (batch_size, epochs, in_channels, times).

        Returns:
        --------
        X_conv : Tensor
            Convolved complex output of shape
            (batch_size, n_freqs, in_channels, times)

        Notes:
        ------
        The multiple epochs are concatenated along the frequency dimension
        after the convolution.

        """

        wavelets = _compute_gaborwavelet(
            tt=self.tt,
            foi=self.foi,
            fwhm=self.fwhm,
            dtype=self.dtype,
            sfreq=self.sfreq,
            scaling=self.scaling
        )
        n_freqs = wavelets.shape[0]

        # If single epoch
        if X.dim() == 3:
            batch_size, in_channels, times = X.shape
            X_conv = F.conv1d(
                # channels to batch element
                X.to(self.dtype).view(-1, 1, times),
                wavelets.unsqueeze(1),
                padding=self.padding,
                stride=self.stride
            )
            # restore channels dimension
            X_conv = X_conv.view(batch_size, in_channels, n_freqs, -1)
            # swap frequency and channels dimension
            X_conv = X_conv.swapaxes(1, 2)

        # If multiple epochs
        elif X.dim() == 4:
            batch_size, n_epochs, in_channels, times = X.shape
            X_conv = F.conv1d(
                # channels to batch element
                X.to(
                    self.dtype).view(
                    batch_size *
                    n_epochs *
                    in_channels,
                    1,
                    times),
                wavelets.unsqueeze(1),
                padding=self.padding,
                stride=self.stride
            )
            X_conv = X_conv.view(
                batch_size, n_epochs, in_channels, n_freqs, -1)
            X_conv = X_conv.permute(0, 3, 2, 1, 4).contiguous()
            n_batch, n_freqs, n_sensors, n_epochs, n_times = X_conv.shape
            X_conv = X_conv.view(
                n_batch,
                n_freqs,
                n_sensors,
                n_epochs *
                n_times)

        return X_conv

    def __repr__(self):
        # This is where you define the representation of your module
        return f"ComplexWavelet(kernel_width_s={self.kernel_width_s}, " \
               f"sfreq={self.sfreq}, n_wavelets={self.n_wavelets}, " \
               f"stride={self.stride}, padding={self.padding}, " \
               f"scaling={self.scaling})"


class RealCovariance(nn.Module):
    """
    Compute the real covariance matrix of the wavelet transformed eeg signals.
    Input shape: (N x F x P x T)
    Output shape: (N x F x P x P)
    """

    def __init__(self,):
        super(RealCovariance, self).__init__()

    def forward(self, X):
        # X: (N x F x P x T)
        assert X.dim() == 4
        cplx_cov = X @ torch.transpose(X, -1, -2).conj() / X.shape[-1]
        return cplx_cov.real


class CrossCovariance(nn.Module):
    """
    Compute the real covariance matrix with cross-frequency interactions of
    the wavelet transformed eeg signals.
    Input shape: (N x F x P x T)
    Output shape: (N x FP x FP)
    """

    def __init__(self,):
        super(CrossCovariance, self).__init__()

    def forward(self, X):
        # X: (N x F x P x T)
        assert X.dim() == 4
        n_batch, n_freqs, n_sensors, n_times = X.shape
        cross_cov = X.view(
            n_batch, n_freqs * n_sensors, n_times
        ) @ X.view(
            n_batch, n_freqs * n_sensors, n_times
        ).transpose(-1, -2).conj() / n_times
        return cross_cov.real


class PW_PLV(nn.Module):
    """Pairwise phase locking value.
    Compute the sensor pairwise phase locking value of the wavelet transformed 
    eeg signals.
    Inspired by https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0193%281999%298%3A4%3C194%3A%3AAID-HBM4%3E3.0.CO%3B2-C # noqa
    PLV between two sensors i and j measures the consistency of the phase lag. It is defined as:
    PLV_{ij} = |<exp(j(\\phi_i - \\phi_j)(t)) >_t|
    """

    def __init__(self, reg=None, n_ch=None) -> None:
        super(PW_PLV, self).__init__()
        if reg is not None:
            self.reg_mat = torch.nn.Parameter(
                torch.eye(n_ch).reshape(1, 1, n_ch, n_ch) * reg,
                requires_grad=False)
        else:
            # register None parameter
            self.register_parameter('reg_mat', None)

    def forward(self, X: Tensor) -> Tensor:
        # X: (N x F x P x T)
        assert X.dim() == 4
        plv_tensor = torch.abs(
            (X / torch.abs(X)) @ torch.transpose(
                X / torch.abs(X), -1, -2).conj()
            / X.shape[-1])
        if self.reg_mat is not None:
            plv_tensor += self.reg_mat
        return plv_tensor


class CrossPW_PLV(nn.Module):
    """Cross-frequency pairwise phase locking value.
    """

    def __init__(self, reg=None, n_ch=None, n_freqs=None) -> None:
        super(CrossPW_PLV, self).__init__()
        if reg is not None:
            n_compo = n_ch * n_freqs
            self.reg_mat = torch.nn.Parameter(
                torch.eye(n_compo).reshape(1, n_compo, n_compo) * reg,
                requires_grad=False)
        else:
            # register None parameter
            self.register_parameter('reg_mat', None)

    def forward(self, X: Tensor) -> Tensor:
        # X: (N x F x P x T)
        assert X.dim() == 4
        n_batch, n_freqs, n_sensors, n_times = X.shape
        X = X.view(n_batch, n_freqs * n_sensors, n_times)

        plv_tensor = torch.abs(
            (X / torch.abs(X)) @ torch.transpose(
                X / torch.abs(X), -1, -2).conj()
            / n_times)
        if self.reg_mat is not None:
            plv_tensor += self.reg_mat
        return plv_tensor


class CombinedPooling(nn.Module):
    """Concatenate along the first axis the features computed by multiple
    pooling layers (Covariance, PLV, etc.)
    """

    def __init__(self, pooling_layers: list) -> None:
        super(CombinedPooling, self).__init__()
        self.pooling_layers = nn.ModuleList(pooling_layers)

    def forward(self, X: Tensor) -> Tensor:
        if isinstance(self.pooling_layers[0], RealCovariance) or isinstance(
                self.pooling_layers[0], PW_PLV):
            return torch.cat([pool(X) for pool in self.pooling_layers], dim=1)
        elif isinstance(self.pooling_layers[0], CrossCovariance) or isinstance(
                self.pooling_layers[0], CrossPW_PLV):
            return torch.cat([pool(X).unsqueeze(1)
                             for pool in self.pooling_layers], dim=1)

    def __getitem__(self, idx):
        return self.pooling_layers[idx]
