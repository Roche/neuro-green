"""
Cross validation utilities used for the experiments.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel
from joblib import delayed
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateFinder
from pl_utils import GreenRegressorLM
from pl_utils import get_train_test_loaders
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from torch.nn import Module
from tqdm import tqdm


def run_one_fold_pl(model: torch.nn.Module,
                    dataset,
                    train_splits: np.ndarray,
                    test_splits: np.ndarray,
                    fold_idx: int,
                    state_dict_cp=None,
                    n_epochs: int = 25,
                    ckpt_path: str = "checkpoints",
                    callbacks=[
                        LearningRateFinder(min_lr=1e-4,
                                           max_lr=1e-2,
                                           num_training_steps=20),
                    ],
                    pl_module=GreenRegressorLM,
                    save_preds=False,
                    batch_size=128,
                    num_workers=8,
                    pl_params: dict = {},
                    test_at_end=True,
                    tta=None,
                    ):
    """
    Function to run one fold of the cross validation using a GREEN model
    implemented using Pytorch Lightning.
    """
    if Path(ckpt_path +
            "/preds.csv").exists() or Path(ckpt_path +
                                           "/y_pred_proba.csv").exists():
        print(f"Fold {fold_idx} already trained")
        return None
    if state_dict_cp is not None:
        model.load_state_dict(state_dict_cp)

    train_indices, test_indices = train_splits[fold_idx], test_splits[fold_idx]

    (train_dataloader,
     test_dataloader,
     final_test_dataloader) = get_train_test_loaders(dataset,
                                                     train_indices,
                                                     test_indices,
                                                     batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     final_val=True)

    model_pl = pl_module(model=model, **pl_params)
    trainer = Trainer(max_epochs=n_epochs,
                      log_every_n_steps=1,
                      callbacks=callbacks,
                      default_root_dir=ckpt_path,
                      enable_checkpointing=False
                      )
    trainer.fit(model=model_pl, train_dataloaders=train_dataloader,
                val_dataloaders=test_dataloader)

    if save_preds:
        trainer.save_checkpoint(ckpt_path + "/checkpoint.ckpt")
        out = trainer.predict(model_pl, dataloaders=final_test_dataloader)
        df_preds = pd.concat(out)
        df_preds['test_indices'] = test_indices
        df_preds.to_pickle(ckpt_path + "/preds.pkl")

    if tta is not None:
        trainer.save_checkpoint(ckpt_path + "/checkpoint.ckpt")
        with torch.no_grad():
            preds_proba = []
            y_true = None
            for i in range(tta):
                test_dataloader.dataset.rng = np.random.default_rng(i)
                # For some reason, calling the dataset again is necessary to
                # reset the random state
                test_dataloader.dataset[0]
                out = pd.concat(trainer.predict(
                    model_pl,
                    dataloaders=test_dataloader
                ))
                preds_proba.append(np.vstack(out['y_pred_proba']))
                if y_true is None:
                    y_true = out['y_true'].to_numpy()
                else:
                    assert np.array_equal(y_true, out['y_true'].to_numpy())

            preds_proba = np.stack(preds_proba, axis=0)
            np.save(ckpt_path + "/y_pred_proba.npy", preds_proba)
            np.save(ckpt_path + "/y_true.npy", y_true)

    elif test_at_end:
        return trainer.test(model_pl, dataloaders=final_test_dataloader)
    else:
        return None


def pl_crossval(
        model: Module,
        dataset,
        n_splits: int = 5,
        n_epochs: int = 25,
        ckpt_prefix: str = 'checkpoints',
        callbacks=[
            LearningRateFinder(min_lr=1e-4,
                               max_lr=1e-2,
                               num_training_steps=20),
        ],
        random_state: int = 0,
        train_splits: np.ndarray = None,
        test_splits: np.ndarray = None,
        pl_module=GreenRegressorLM,
        save_preds=False,
        batch_size=128,
        num_workers=8,
        pl_params: dict = {'weight_decay': 1e-3},
        test_at_end: bool = True,
        tta: int = None):
    """Cross validation of SPDNet in Lightning.

    Parameters
    ----------
    model : Module
        Pytorch model used for the cross validation
    dataset : Dataset
        EpochDataset containing the EEG data
    n_splits : int, optional
        Number of splits for the CV, by default 5
    n_jobs : int, optional
        Number of parallel jobs, by default 1
    n_epochs : int, optional
        Number of train epochs for the model, by default 25
    ckpt_prefix : str, optional
        Prefix of the checkpoint, by default 'checkpoints'
    callbacks : list, optional
        List of callbacks for the Lightning Trainer, by default [
        LitProgressBar(),
        LearningRateFinder(min_lr=1e-4,
                           max_lr=1e-2,
                           num_training_steps=20),
    ]
    random_state : int, optional
        random state for reproducibility, by default 0
    train_splits : np.ndarray, optional
        Predefined train splits, by default None
    test_splits : np.ndarray, optional
        Predefined test splits, by default None
    pl_module : Module, optional
        Lightning module, by default GreenRegressorLM
    save_preds : bool, optional
        Whether to save the predictions of the model, by default False
    batch_size : int, optional
        Batch size, by default 128
    num_workers : int, optional
        Number of workers for the dataloader, by default 8
    pl_params : dict, optional
        Parameters for the Lightning module, by default {'weight_decay': 1e-3}
    test_at_end : bool, optional
        Whether to test the model at the end of the training, by default True

    Returns
    -------
    out : list
        List of the results (score) of the cross validation
    """
    if train_splits is None or test_splits is None:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_splits = list()
        test_splits = list()

        for train, test in cv.split(np.arange(len(dataset))):
            train_splits.append(train)
            test_splits.append(test)

    state_dict_cp = deepcopy(model.state_dict())

    iter_range = reversed(range(len(train_splits)))

    out = [run_one_fold_pl(
        model,
        dataset,
        train_splits,
        test_splits,
        fold_idx=fold_idx,
        state_dict_cp=state_dict_cp,
        n_epochs=n_epochs,
        ckpt_path=f"./{ckpt_prefix}/fold{fold_idx}",
        callbacks=callbacks,
        pl_module=pl_module,
        pl_params=pl_params,
        save_preds=save_preds,
        batch_size=batch_size,
        num_workers=num_workers,
        test_at_end=test_at_end,
        tta=tta,
    ) for fold_idx in iter_range]
    return out, test_splits


def run_one_fold_sk(model,
                    X,
                    y,
                    train_splits,
                    test_splits,
                    fold_idx):
    """
    Function to run one fold of the cross validation using a sklearn model.
    """

    train_indices, test_indices = train_splits[fold_idx], test_splits[fold_idx]
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    else:
        X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    df_out = pd.DataFrame(dict(
        y_pred=y_pred,
        y_test=y_test,
        test_indices=test_indices
    ))
    df_out['fold'] = fold_idx
    return df_out


def sk_crossval(
        model: BaseEstimator,
        X: pd.DataFrame,
        y: np.ndarray,
        n_splits: int = 5,
        n_jobs: int = 1,
        random_state: int = 0,
        train_splits: np.ndarray = None,
        test_splits: np.ndarray = None,
) -> list:
    """
    Parallelized cross validation for sklearn models.

    Parameters
    ----------
    model : BaseEstimator
        sklearn model
    X : pd.DataFrame
        Dataframe containing the data (columns represent frequencies and rows
        subjects)
    y : np.ndarray
        Labels
    n_splits : int, optional
        Number of splits for the CV, by default 5
    n_jobs : int, optional
        Number of jobs to run in parallel, by default 1
    random_state : int, optional
        random state for reproducibility, by default 0
    train_splits : np.ndarray, optional
        Predefined train splits, by default None
    test_splits : np.ndarray, optional
        Predefined test splits, by default Nones

    Returns
    -------
    list
        List of the results (score) of the cross validation
    """
    if train_splits is None or test_splits is None:

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_splits = list()
        test_splits = list()

        for train, test in cv.split(np.arange(len(y))):
            train_splits.append(train)
            test_splits.append(test)

    out = Parallel(n_jobs=n_jobs)(delayed(run_one_fold_sk)(
        model,
        X,
        y,
        train_splits,
        test_splits,
        fold_idx=fold_idx,
    ) for fold_idx in tqdm(range(len(train_splits))))
    return pd.concat(out)
