# GREEN architecture
![CI](https://github.com/Roche/neuro-green/actions/workflows/lint_and_test.yaml/badge.svg)
---

## About the architecture
The model is a deep learning architecture designed for EEG data that combines wavelet transforms and Riemannian geometry. It is capable of learning from raw EEG data and can be applied to both classification and regression tasks. The model is composed of the following layers:
It is based on the following layers:
 - Convolution: Uses complex-valued Gabor wavelets with parameters that are learned during training. 
 - Pooling: Derives features from the wavelet-transformed signal, such as covariance matrices.
 - Shrinkage layer: applies [shrinkage](https://scikit-learn.org/1.5/modules/covariance.html#basic-shrinkage) to the covariance matrices.
 - Riemannian Layers: Applies transformations to the matrices, leveraging the geometry of the Symmetric Positive Definite (SPD) manifold.
 - Fully Connected Layers: Standard fully connected layers for final processing.

![alt text](assets/concept_figure.png)


## Getting started 

```
pip install -e .
```

## Dependencies 
``` 
scikit-learn
torch
geotorch
lightning
mne
```

## Examples
Examples illustrating how to train the presented model can be found in the `green/research_code` folder. The notebook `example.ipynb` shows how to train the model on raw EEG data. And the notebook `example_wo_wav.ipynb` shows how to train a submodel that uses covariance matrices as input. 

In addition, being pure PyTorch, the GREEN model can easily be integrated to [`braindecode`](https://braindecode.org/stable/index.html) routines. 

```python
import torch
from braindecode import EEGRegressor
from green.wavelet_layers import RealCovariance
from green.research_code.pl_utils import get_green

green_model = get_green(
	n_freqs=5,                    # Learning 5 wavelets
	n_ch=22,                      # EEG data with 22 channels
	sfreq=100,   			      # Sampling frequency of 100 Hz
	dropout=0.5,		          # Dropout rate of 0.5 in FC layers
	hidden_dim=[100],             # Use 100 units in the hidden layer
	pool_layer=RealCovariance(),  # Compute covariance after wavelet transform
	bi_out=[20],    		      # Use a BiMap layer outputing a 20x20 matrix
	out_dim=1, 				      # Output dimension of 1, for regression
)

device = "cuda" if torch.cuda.is_available() else "cpu"
EarlyStopping(monitor="valid_loss", patience=10, load_best=True)
clf = EEGRegressor(
	module=green_model,
	criterion=torch.nn.CrossEntropyLoss,
	optimizer=torch.optim.AdamW,
	device=device,
	callbacks=[],                  # Callbacks can be added here, e.g. EarlyStopping
)
```

## Citation
When using our code, please cite the reference article:

``` bibtex
@article {paillard_2024_green,
	author = {Paillard, Joseph and Hipp, Joerg F and Engemann, Denis A},
	title = {GREEN: a lightweight architecture using learnable wavelets and Riemannian geometry for biomarker exploration},
	year = {2024},
	doi = {10.1101/2024.05.14.594142},
	URL = {https://www.biorxiv.org/content/early/2024/05/14/2024.05.14.594142},
	journal = {bioRxiv}
}
```