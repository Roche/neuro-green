import numpy as np
import torch

from green.data_utils import EpochsDataset
from green.wavelet_layers import PW_PLV
from green.wavelet_layers import CombinedPooling
from green.wavelet_layers import CrossCovariance
from green.wavelet_layers import CrossPW_PLV
from green.wavelet_layers import RealCovariance
from green.wavelet_layers import WaveletConv


def test_wavelet_conv(dummy_mne_epochs):
    layer = WaveletConv(
        kernel_width_s=0.5,
        sfreq=100,
        foi_init=np.array([1, 2, 4]),
        fwhm_init=np.array([1, .5, .25]),
        padding='valid',
        dtype=torch.complex64,
        stride=1,
        scaling='oct'
    )
    dataset = EpochsDataset(
        epochs=dummy_mne_epochs,
        targets=[0] * 10,
        n_epochs=2,
        subjects=['subject1'] * 10,
        shuffle_first_epoch=True,
        shuffle=True,
    )
    input = dataset[0][0].repeat(2, 1, 1, 1)
    out = layer(input)
    # (n_batch, n_epochs, n_channels, n_times)
    # time dimension is reduced by the convolution + concatenated epochs
    assert out.shape[:-1] == (2, 3, 3)
    assert out.dtype == torch.complex64

    # Test pooling lauers
    pool_1 = CombinedPooling(
        pooling_layers=[RealCovariance(), PW_PLV(reg=1e-3, n_ch=3)]
    )
    out_1 = pool_1(out)
    # N x PF X C x C
    assert out_1.shape == (2, 2 * 3, 3, 3)

    pool_2 = CombinedPooling(
        pooling_layers=[
            CrossCovariance(),
            CrossPW_PLV(reg=1e-3, n_ch=3, n_freqs=3)
        ])
    out_2 = pool_2(out)
    # N x P x FC x FC
    assert out_2.shape == (2, 2, 3 * 3, 3 * 3)
