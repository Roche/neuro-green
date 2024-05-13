import numpy as np
import torch
from torch.utils.data import Dataset


class EpochsDataset(Dataset):
    def __init__(self,
                 epochs: list,
                 targets: list,
                 n_epochs: int,
                 subjects: list,
                 padding='repeat',
                 shuffle_first_epoch: bool = False,
                 shuffle: bool = False,
                 random_state: int = 0,
                 transform_y=lambda x: x,
                 transform_x=lambda x: torch.Tensor(x * 1e6 / 8),
                 age=None,):
        """
        X: epochs (n_epochs, n_channels, n_times)
        y: target


        Parameters
        ----------
        epochs : list of mne.Epochs
            List of mne Epochs objects.
        targets : list
            Labels.
        n_epochs : int
            Number of epochs sampled per subject.
        subjects : list
            Names of the subjects.
        padding : int or str, optional
            Method used for padding when the number of epochs available is
            less than n_epochs.
        shuffle_first_epoch : bool, optional
            Whether to shuffle the first epoch or not. Default is False.
        random_state : int, optional
            Random state. Default is 0.
        transform_y : function, optional
            Transformation applied on the fly to the targets. Default is
            lambda x: (x - 40) / 20.
        transform_x : function, optional
            Transformation applied on the fly to the inputs. Default is
            lambda x: torch.Tensor(x * 1e6 / 8).
        shuffle : bool, optional
            Whether to shuffle the epochs or not. Default is False.
        age : list, optional
            Age of the subjects. Default is None.
        """
        self.epochs = epochs
        self.targets = targets
        self.n_epochs = n_epochs
        self.subjects = subjects
        self.transform_y = transform_y
        self.transform_x = transform_x
        self.padding = padding
        self.shuffle_first_epoch = shuffle_first_epoch
        self.rng = np.random.default_rng(random_state)
        self.shuffle = shuffle
        self.age = age

    def __getitem__(self, idx):

        # shuffle the start index of the epochs.
        # TODO: consider using a random sampling mixing all the epochs instead
        if self.shuffle_first_epoch and (
                len(self.epochs[idx]) > self.n_epochs):
            start_idx = self.rng.choice(
                np.arange(len(self.epochs[idx]) - self.n_epochs)
            )
        else:
            start_idx = 0

        # get the epochs and the target
        X_orig = self.epochs[idx][start_idx: start_idx +
                                  self.n_epochs].get_data()
        y = self.targets[idx]

        if self.shuffle:
            # Sample with replacement if n_epochs > len(epochs)
            sample_idx = self.rng.choice(
                np.arange(X_orig.shape[0]),
                replace=self.n_epochs > X_orig.shape[0],
                size=self.n_epochs
            )
            X = X_orig[sample_idx]
        else:
            # padding
            padding_size = self.n_epochs - X_orig.shape[0]
            if (padding_size > 0) and (self.padding is not None):
                if self.padding == 'zero':
                    X = np.pad(
                        X_orig, ((0, padding_size), (0, 0), (0, 0)),
                        mode='constant')

                elif self.padding == 'repeat':
                    repeated_ids = self.rng.choice(
                        np.arange(X_orig.shape[0]),
                        size=padding_size,
                        replace=True
                    )
                    X = np.concatenate([X_orig, X_orig[repeated_ids]], axis=0)
            else:
                X = X_orig

        y = self.transform_y(y)
        X = self.transform_x(X)
        if self.age is not None:
            return X, self.age[idx], y
        else:
            return X, y

    def __len__(self):
        return len(self.epochs)

    def __repr__(self) -> str:
        epoch_ex = self.epochs[0]
        num_channels = epoch_ex.get_data().shape[1]
        sfreq = epoch_ex.info['sfreq']
        epo_time_s = epoch_ex.times[-1] - epoch_ex.times[0]

        params_str = f"len: {len(self)}\n" \
                     f"n_epochs/sample: {self.n_epochs}\n" \
                     f"num_channels/sample: {num_channels}\n" \
                     f"sampling frequency: {sfreq}\n" \
                     f"epoch duration (s): {epo_time_s}\n" \
                     f"padding: {self.padding}\n" \
                     f"shuffle: {self.shuffle}\n" \
                     f"random_state: {self.rng}\n" \
                     f"use age: {self.age}"
        return f"EpochsDataset\n=====================\n{params_str}"
