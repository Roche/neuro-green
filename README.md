# GREEN architecture (Gabor Riemann EEGNet)
![CI](https://github.com/Roche/neuro-green/actions/workflows/lint_and_test.yaml/badge.svg)
[![DOI](https://zenodo.org/badge/795238657.svg)](https://doi.org/10.5281/zenodo.14597453)
---

## About the architecture

GREEN is a deep learning architecture designed for EEG data that combines wavelet transforms and Riemannian geometry.

The model (see our [paper](https://doi.org/10.1016/j.patter.2025.101182) for details) is composed of the following layers:

 - Convolution: Uses complex-valued Gabor wavelets with parameters that are learned during training. 

 - Pooling: Derives features from the wavelet-transformed signal, such as covariance matrices.

 - Shrinkage layer: applies [shrinkage](https://scikit-learn.org/1.5/modules/covariance.html#basic-shrinkage) to the covariance matrices.

 - Riemannian Layers: Applies transformations to the matrices, leveraging the geometry of the Symmetric Positive Definite (SPD) manifold.

 - Fully Connected Layers: Standard fully connected layers for final processing.

![alt text](assets/concept_figure.png)

To dive into the background, check out our [paper](https://doi.org/10.1016/j.patter.2025.101182) published in Patterns (Cell Press).

## Getting started
Clone the repository and install locally.

```
pip install -e .
```

## Dependencies 

You will need the following dependencies to get most out of GREEN.

```
scikit-learn
torch
geotorch
lightning
mne
```

## Examples

Examples illustrating how to train the presented model can be found in the [`green/research_code`](https://github.com/Roche/neuro-green/tree/main/green/research_code) folder. The notebook [`example.ipynb`](https://github.com/Roche/neuro-green/blob/main/green/research_code/example.ipynb) shows how to train the model on raw EEG data. And the notebook [`example_wo_wav.ipynb`](https://github.com/Roche/neuro-green/blob/main/green/research_code/example_wo_wav.ipynb) shows how to train a submodel that uses covariance matrices as input. 

In addition, being pure PyTorch, the GREEN model can easily be integrated to [`braindecode`](https://braindecode.org/stable/index.html) routines. 

```python
import torch
from braindecode import EEGRegressor
from green.wavelet_layers import RealCovariance
from green.research_code.pl_utils import get_green

green_model = get_green(
	n_freqs=5,	# Learning 5 wavelets
	n_ch=22,	# EEG data with 22 channels
	sfreq=100,	# Sampling frequency of 100 Hz
	dropout=0.5,	# Dropout rate of 0.5 in FC layers
	hidden_dim=[100],	# Use 100 units in the hidden layer
	pool_layer=RealCovariance(),	# Compute covariance after wavelet transform
	bi_out=[20],	# Use a BiMap layer outputing a 20x20 matrix
	out_dim=1,	# Output dimension of 1, for regression
)

device = "cuda" if torch.cuda.is_available() else "cpu"
EarlyStopping(monitor="valid_loss", patience=10, load_best=True)
clf = EEGRegressor(
	module=green_model,
	criterion=torch.nn.CrossEntropyLoss,
	optimizer=torch.optim.AdamW,
	device=device,
	callbacks=[],	# Callbacks can be added here, e.g. EarlyStopping
)
```

## Citation
When using our code, please cite the reference article:

``` bibtex

@article{paillard2025,
    author = {Paillard, Joseph and Hipp, J{\"o}rg F. and Engemann, Denis A.},
    title = {GREEN: A lightweight architecture using learnable wavelets and Riemannian geometry for biomarker exploration with EEG signals},
    doi = {10.1016/j.patter.2025.101182},
    url = {https://doi.org/10.1016/j.patter.2025.101182},
    journal = {Patterns},
    publisher = {Elsevier},
    isbn = {2666-3899}
}
```

## Contributing

We currently do not accept contributions.
