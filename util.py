import numpy as np

__all__ = ["sample_twod", "quantile"]


# ------------------------------
# Sample from a 2d histogram
# ------------------------------
def sample_twod(X, Y, Z, n_sample=1000):

    sflat = Z.flatten()
    sind = np.arange(len(sflat))
    inds = np.random.choice(sind, size=n_sample,
                            p=sflat / np.nansum(sflat))
    # TODO: check x, y order here
    N = len(np.squeeze(Y))
    xx = inds // N
    yy = np.mod(inds, N)

    y = np.squeeze(Y)[yy]
    x = np.squeeze(X)[xx]
    return x, y


def quantile(xarr, q, weights=None):
    """Compute (weighted) quantiles from an input set of samples.

    :param x: `~numpy.darray` with shape (nvar, nsamples)
        The input array to compute quantiles of.

    :param q: list of quantiles, from [0., 1.]

    :param weights: shape (nsamples)

    :returns quants: ndarray of shape (nvar, nq)
        The quantiles of each varaible.
    """
    qq = [_quantile(x, q, weights=weights) for x in xarr]
    return np.array(qq)


def _quantile(x, q, weights=None):
    """Compute (weighted) quantiles from an input set of samples.

    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.

    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.

    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.

    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.

    """
    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q))
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles
