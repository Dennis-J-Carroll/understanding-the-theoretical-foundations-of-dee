import numpy as np
import torch
from sklearn.datasets import make_moons


def get_moons_data(n_samples=400, noise=0.2, seed=0, test_frac=0.3):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_samples)
    n_test = int(n_samples * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    return (
        torch.from_numpy(X[train_idx]),
        torch.from_numpy(y[train_idx]),
        torch.from_numpy(X[test_idx]),
        torch.from_numpy(y[test_idx]),
    )
