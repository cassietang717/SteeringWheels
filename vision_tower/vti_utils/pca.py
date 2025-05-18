import torch
import torch.nn as nn
import torch.nn.functional as F


def svd_flip(u, v):
    """
    For each component, the column of U is flipped (if needed) to ensure 
    the entry with the largest absolute value in each column of U is positive, 
    and the corresponding row of Vh is flipped to maintain consistency.
    """
    # u: [T, N, K], vh: [T, K, D]
    # columns of u, rows of vh

    # [T, K]
    # for each token t and component k, max row index i with the maximum absolute value
    max_abs_cols = torch.argmax(torch.abs(u), dim=1)
    # [T, K, 1]
    max_abs_cols = max_abs_cols.unsqueeze(-1)  # just to match the dimensions for gather, but not necessary to expand further
    
    #  sign of the max-abs element in u[t, :, k] for each token t and component k
    signs = torch.sign(torch.gather(u, 1, max_abs_cols)) # [T, K, 1]

    # change the sign of the columns of u
    u *= signs # [T, N, K]
    # (T, K, 1) => change the sign of rows of vh 
    v *= signs.view(v.shape[0], -1, 1) # [T, K, D]
    return u, v


class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        # [N, D]
        if X.ndim == 2:
            n, d = X.size()
            X = X.unsqueeze(0)
        # [T, N, D]
        elif X.ndim == 3:
            _, n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        
        # [T, 1, D]
        self.register_buffer("mean_", X.mean(1, keepdim=True))
        Z = X - self.mean_ # center

        # [T, N, D], [D, D], [T, D, D]
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        # [T, d, D]
        self.register_buffer("components_", Vt[:, :d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        # [T, N, D] @ [T, D, d] => [T, N, d]
        return torch.matmul(X - self.mean_, self.components_.transpose(-2, -1))

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        # [T, N, d] @ [T, d, D] => [T, N, D]
        return torch.matmul(Y, self.components_) + self.mean_
