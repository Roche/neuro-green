import numpy as np
import torch
from torch import Tensor
from torch import nn


def _modify_eigenvalues(X: Tensor, function) -> Tensor:
    """Apply function to the eigenvalues of a covariance matrix

    Parameters
    ----------
    X : Tensor
        Covariances matrices considered. Shape N x F x C x C
    function : _type_
        Function to apply to the eigenvalues

    Returns
    -------
    Tensor
        Set of covariance matrices with modified eigenvalues
    """
    D, Q = torch.linalg.eigh(X)
    D_out = function(D)
    X_out = Q @ torch.diag_embed(D_out) @ torch.transpose(Q, -2, -1)
    return X_out


def mean_logeuclid(X, weights=None, reg=1e-4):
    """
    https://epubs.siam.org/doi/abs/10.1137/050637996
    """
    def _rec_log(X,):
        return torch.log(torch.max(X, torch.ones_like(X) * reg))

    if weights is not None:
        out = _modify_eigenvalues(
            torch.sum(_modify_eigenvalues(X, _rec_log) * weights,
                      dim=0) / torch.sum(weights, dim=0),
            torch.exp)
    else:
        out = _modify_eigenvalues(
            torch.mean(_modify_eigenvalues(X, _rec_log), dim=0),
            torch.exp)
    return out


class Shrinkage(nn.Module, ):
    def __init__(self,
                 size: int,
                 n_freqs: int = 1,
                 init_shrinkage=None,
                 learnable=False):
        """
        Applies shrinkage to (possibly) ill-conditioned matrices.

        Parameters:
        -----------
        size : int
            The size of the matrices.
        n_freqs: int, optional
            The number of frequency bands. Default is 1.
        init_shrinkage: float, optional
            The initial shrinkage value. If None, it is set to 0. Default
            is None.
        learnable: bool, optional
            Whether the shrinkage parameter is learnable. Default is False.
        """
        super().__init__()
        self.size = size
        self.n_freqs = n_freqs
        self.learnable = learnable
        self.init_shrinkage = init_shrinkage

        if n_freqs is not None:
            if init_shrinkage is not None:
                # Transform the shrinkage value using sigmoid function to
                # ensure it is between 0 and 1
                alpha = torch.stack(([
                    torch.tensor(init_shrinkage) for _ in range(n_freqs)]),
                    dim=0
                ).reshape(1, n_freqs, 1, 1)
            else:
                alpha = torch.zeros(1, n_freqs, 1, 1)
            shrink_mat = torch.stack([
                torch.eye(size) for _ in range(n_freqs)
            ], dim=0).unsqueeze(0)
        else:
            if init_shrinkage is not None:
                alpha = torch.tensor(init_shrinkage).reshape(1, 1, 1)
            else:
                alpha = torch.tensor(0.).reshape(1, 1, 1)

            shrink_mat = torch.eye(size).unsqueeze(0)

        self.shrinkage = nn.Parameter(alpha, requires_grad=learnable)

        reg_noise = torch.randn(shrink_mat.shape) / 10

        self.shrink_mat = nn.Parameter(
            shrink_mat + reg_noise,
            requires_grad=False)

    def forward(self, X):
        trace = torch.sum(
            torch.diagonal(X, dim1=-2, dim2=-1).unsqueeze(-1),
            dim=-2,
            keepdims=True
        )
        mu = trace / self.size

        alpha = torch.sigmoid(self.shrinkage)
        out = (1 - alpha) * X + alpha * mu * self.shrink_mat
        return out

    def __repr__(self):
        return f"LedoitWold(n_freqs={self.n_freqs}, " \
               f"init_shrinkage={self.init_shrinkage}, " \
               f"learnable={self.learnable})"


class BiMap(nn.Module):
    """
    Applies BiLinear transformation to the data: $y = W x W^T$

    Note: weight matrix is not necessarly orthogonal.
    The parametrization takes place in the Covceptron class.

    """

    def __init__(self, d_in: int, d_out: int, n_freqs: int = 1):
        """
        Parameters
        ----------
        d_in : int
            Dimension of input covariance matrices
        d_out : int
            Rows of the weights matrix. Dim of the output PD matrix.
        n_freqs : int, optional
            Number of frequencies. Similar to channels in images.

        Note
        ----
        To add semi-orthogonal constaint for the weight matrix, use the
        reparametrization trick:
        ```
        >>> import geotorch
        >>> geotorch.orthogonal(bimap, 'weight')
        ```

        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_freqs = n_freqs
        if n_freqs is not None:
            self.weight = nn.Parameter(torch.empty(
                self.n_freqs, self.d_out, self.d_in))
        else:
            self.weight = nn.Parameter(torch.empty(self.d_out, self.d_in))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, X):
        h = self.weight @ X @ torch.transpose(self.weight, -2, -1)
        return h

    def __repr__(self):
        return f"BiMap(d_in={self.d_in}, d_out={self.d_out}, " \
               f"n_freqs={self.n_freqs}"


class ReEig(nn.Module):
    """
    Non-linearity that forces the eigenvalues to be greater or equal to a
    threshold $\\epsilon$

    Applies $y = Q \\max(\\epsilon I, D) Q^T, \\quad$ where $x = QDQ^T$

    By regularizing the eigenvalues, this layer should solve rank defficiency
    problems and ensure that the computation performed in the `LogEig` layer is
    valid.
    """

    def __init__(self, epsilon: float = 1e-6):
        """
        Parameters
        ----------
        epsilon : float, optional
            Eigenvalue regularization parameter, by default 1e-6
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, X: Tensor):
        D, Q = torch.linalg.eigh(X)
        D_out = torch.max(D, torch.ones_like(D) * self.epsilon)
        return Q @ torch.diag_embed(D_out) @ torch.transpose(Q, -2, -1)


class RunningMeanCov():
    def __init__(self, size: int, n_freqs: int, momentum: float):
        """Running log-Euclidean mean for covariance matrices

        Parameters
        ----------
        size : int
            Shape of covariance matrices
        n_freqs : int
            Number of frequencies, similar to channels for images
        momentum : float
            Momentum for the update of the running mean.
        """
        self.value = torch.stack(
            [torch.eye(size) for _ in range(n_freqs)],
            dim=0
        )
        self.n_freqs = n_freqs
        self.size = size
        self.momentum = momentum
        self.weights = torch.Tensor(
            [1, 1 - self.momentum]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def update(self, X):
        X_mean = mean_logeuclid(X)
        self.value = mean_logeuclid(
            torch.stack([self.value, X_mean], dim=0),
            weights=self.weights
        )


class LogMap(nn.Module):
    """
    Applies matrix logarithm and vectorize by taking the upper triangle.

    $y = Q \\log(D) Q^T \\quad$ where $x = QDQ^T$
    """

    def __init__(
            self,
            ref: str = 'identity',
            size: int = None,
            n_freqs: int = None,
            momentum: float = None,
            reg=1e-6):
        """Log-Euclidean layer

        Parameters
        ----------
        ref : str, optional
            Method for computing the reference point for the projection to
            tangent space , by default 'identify'
        size : int, optional
            see RunningMeanCov, by default None
        n_freqs : int, optional
            see RunningMeanCov, by default None
        momentum : float, optional
            see RunningMeanCov, by default None
        reg : float, optional
            If not None, compute a ReEig forward before applying the log using
            the given value, by default 1e-6. Merging the two operations in a
            single layer is more efficient. Also provides numerical stability
            for the log operation.
        """
        super().__init__()
        self.ref = ref
        self.momentum = momentum
        self.n_freqs = n_freqs
        self.size = size
        if self.ref == 'logeuclid':
            if n_freqs is not None:
                self.running_mean = nn.Parameter(
                    torch.stack([torch.eye(size) for _ in range(n_freqs)],
                                dim=0),
                    requires_grad=False
                )
                self.weights = nn.Parameter(torch.Tensor(
                    [1, 1 - self.momentum]).reshape(2, 1, 1, 1),
                    requires_grad=False)
            else:
                self.running_mean = nn.Parameter(torch.eye(size),
                                                 requires_grad=False)
                self.weights = nn.Parameter(torch.Tensor(
                    [1, 1 - self.momentum]).reshape(2, 1, 1),
                    requires_grad=False)

        self.reg = reg

    def forward(self, X: Tensor):
        # Replace negative eigenvalue by reg
        def rec_log(X):
            return torch.log(torch.max(X, torch.ones_like(X) * self.reg))

        # Matrix log
        X_out = _modify_eigenvalues(X, rec_log)

        if self.ref == 'identity':
            log_C_ref = 0

        elif self.ref == 'logeuclid':
            if self.training:
                with torch.no_grad():
                    batch_mean = mean_logeuclid(X)
                    self.running_mean.data = mean_logeuclid(
                        torch.stack([self.running_mean, batch_mean], dim=0),
                        weights=self.weights
                    )
            log_C_ref = torch.Tensor(
                _modify_eigenvalues(self.running_mean, rec_log)
            )
        X_out -= log_C_ref
        return X_out

    def __repr__(self):
        # This is where you define the representation of your module
        return f"LogEig(ref={self.ref}, " \
               f"reg={self.reg}, n_freqs={self.n_freqs}, size={self.size}"
