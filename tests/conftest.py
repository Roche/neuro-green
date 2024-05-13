import mne
import numpy as np
import pytest


def make_one_dummy_epoch():
    info = mne.create_info(
        ch_names=['Oz', 'Pz', 'Cz'],
        sfreq=100,
        ch_types='eeg'
    )
    data = np.random.randn(2, 3, 200)
    epoch = mne.EpochsArray(data, info)
    return epoch


@pytest.fixture
def dummy_mne_epochs():
    epochs = [make_one_dummy_epoch() for _ in range(10)]
    return epochs


@pytest.fixture
def dummy_spd_ill_cond():
    # N x F x C
    U = np.exp(np.random.randn(2, 4, 3))
    U[:, :, 0] = -1e-6
    diag = U[..., np.newaxis] * np.eye(3)
    V = np.random.randn(2, 4, 3, 3)
    spd_ill = V @ diag @ np.transpose(V, (0, 1, 3, 2))
    return spd_ill
