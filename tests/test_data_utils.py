from green.data_utils import EpochsDataset


def test_epochs_dataset(dummy_mne_epochs):
    dataset = EpochsDataset(
        epochs=dummy_mne_epochs,
        targets=[0] * 10,
        n_epochs=3,
        subjects=['subject1'] * 10,
        shuffle_first_epoch=True,
        shuffle=True,

    )
    assert len(dataset) == 10
    X, _ = dataset[0]
    assert X.shape == (3, 3, 200)
