import numpy as np
from astropy.cosmology import Planck15 as cosmo


def schechter(logl, logphi, loglstar, alpha, l_min=None):
    """
    Generate a Schechter function (in dlogl).
    """
    phi = ((10**logphi) * np.log(10) * 10**((logl - loglstar) * (alpha + 1)) * np.exp(-10**(logl - loglstar)))
    return phi


class EvolvingSchecter:

    def __init__(self, zref=14, order=2):
        self.zref = zref
        self.order = 2

        # determines mapping from theta vector to parameters
        self._phi_index = [0, 1, 2]
        self._lstar_index = [3, 4, 5]
        self._alpha_index = [6]

    def set_parameters(self, q):
        self._phis = q[self._phi_index]
        self._lstars = q[self._lstar_index]
        self._alphas = q[self._alpha_index]

    def set_redshift(self, z):
        zz = z - self.zref
        self.phi = np.dot(np.vander(zz, len(self._phis)), self._phis, )
        self.lstar = np.dot(np.vander(zz, len(self._lstars)), self._lstars)
        self.alpha = np.dot(np.vander(zz, len(self._alphas)), self._alphas)

    def evaluate(self, L, z):
        self.set_redshift(z)
        x = (L/self.lstar)
        return self.phi * z**self.alpha * np.exp(-x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def effective_volume(zgrid, lgrid, omega, mag_50=30, dc=1):
    """compute this on a grid of z and Muv
    """
    volume = omega * cosmo.differential_comoving_volume(zgrid).value
    muv = lgrid + cosmo.distmod(zgrid).value
    completeness_function = sigmoid((mag_50-mag) / dc)  # fake completeness function
    selection_function = completeness_function * 1.0

    return selection_function * volume, zgrid, lgrid


def lnlike(q, data, effective_volume):
    veff, zgrid, Muv_grid = effective_volume
    lgrid = 10**(-0.4 * Muv_grid)  # units are now absolute maggies
    s = EvolvingSchecter()
    s.set_parameters(q)

    # if data likelihoods are evluated on the same grid
    schecter = s.evaluate(lgrid, zgrid)
    N_theta = integrate(np.ones_like(schecter), schecter, veff)
    lnlike = np.zeros(len(data))
    for i, d in enumerate(data):
        like = integrate(d.probability, schecter, veff)
        lnlike[i] = np.log(like)

    return np.sum(lnlike) - np.log(N_theta)


if __name__ == "__main__":
    zmin, zmax, nz = 12, 16, 1000
    loglmin, loglmax, nl = 15/2.5, 22/2.5, 1000
    zgrid = np.linspace(zmin, zmax, nz)[None, :]
    loglgrid = np.linspace(loglmin, loglmax, nl)[:, None]
    lgrid = 10**loglgrid

    q_true = np.array([0.0, 0.0, 1e-3, 0, 0, 10**(18 / 2.5), -1.5])
    s = EvolvingSchecter()
    s.set_parameters(q_true)
    # schecter function evaluated on a grid of z and L
    schecter = s.evaluate(lgrid, zgrid[0])
