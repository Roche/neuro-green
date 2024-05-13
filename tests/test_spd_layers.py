import geotorch
import torch
from torch import nn

from green.spd_layers import BiMap
from green.spd_layers import LogMap
from green.spd_layers import ReEig
from green.spd_layers import Shrinkage


def test_spd_base_layers(dummy_spd_ill_cond):

    bimap = BiMap(d_in=3, d_out=2, n_freqs=4)
    geotorch.orthogonal(bimap, 'weight')
    layers = nn.Sequential(*[
        Shrinkage(size=3, n_freqs=4, learnable=True, init_shrinkage=1e-3),
        ReEig(),
        bimap,
        LogMap(ref='logeuclid', size=2, n_freqs=4, momentum=0.9),
    ])
    input_X = torch.Tensor(dummy_spd_ill_cond)
    out = layers(input_X)
    assert out.shape == (2, 4, 2, 2)

    # Make sure that the weights of the BiMap are orthogonal
    W = layers[2].weight
    for p in W @ W.transpose(1, 2):
        assert torch.allclose(p, torch.eye(p.shape[0]), atol=1e-6)
