#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
import numpy as np
import matplotlib.pyplot as pl

from lf import create_parser
from lf import EvolvingSchechter, sample_twod
from lf import effective_volume, lnlike
from lf import lum_to_mag, mag_to_lum, arcmin

maggies_to_nJy = 3631e9


class DataSamples:
    """Holds posterior samples for logL and zred for all objects being used to
    constrain the LF.  Optionally include flux samples to incorporate k-correction effects or other
    """

    def __init__(self, objects, n_samples=1e4):
        self.n_samples = n_samples
        dtype = self.get_dtype(n_samples)
        self.all_samples = np.zeros(len(objects), dtype=dtype)
        for i, obj in enumerate(objects):
            for k in obj.keys():
                self.all_samples[i][k] = obj[k]

    def add_objects(self, objects):
        dtype = self.get_dtype(self.n_samples)
        new = np.zeros(len(objects), dtype=dtype)
        for i, obj in enumerate(objects):
            for k in obj.keys():
                new[i][k] = obj[k]
        self.all_samples = np.concatenate(self.all_samples, new)

    def get_dtype(self, n_samples):
        truecols = [("logl_true", float), ("zred_true", float)]
        sample_cols = [("logl_samples", float, (n_samples,)),
                       ("zred_samples", float, (n_samples,)),
                       ("sample_weight", float, (n_samples,))]
        flux_cols = [("flux_samples", float, (n_samples,))]
        return np.dtype([("id", int)] + sample_cols + flux_cols + truecols)


def make_mock(loglgrid, zgrid, omega,
              q_true,
              n_sample=100,
              noisy=False,
              sigma_logz=0.1, sigma_flux=1/maggies_to_nJy,
              completeness_kwargs={},
              selection_kwargs={}):

    lf = EvolvingSchechter()
    lf.set_parameters(q_true)
    veff = effective_volume(loglgrid, zgrid, omega,
                            completeness_kwargs=completeness_kwargs,
                            selection_kwargs=selection_kwargs,
                            as_interpolator=True)

    dN, dV = lf.n_effective(veff)
    N_bar = dN.sum()
    N = np.random.poisson(N_bar, size=1)[0]
    logl_true, zred_true = sample_twod(loglgrid, zgrid, dN,
                                       n_sample=N)

    data = []
    for logl, zred in zip(logl_true, zred_true):
        l_s, z_s = sample_mock_noise(logl, zred, n_samples=1000)
        obj = dict(logl_true=logl, zred_true=zred,
                   logl_samples=l_s, zred_samples=z_s)
        data.append(obj)
    alldata = DataSamples(data, n_samples=1000)

    return alldata, veff


def sample_mock_noise(logl, zred, n_samples=1000,
                      sigma_flux=1/maggies_to_nJy,  # 1nJy limit
                      sigma_logz=0.1):
    # sample the p(z) distribution
    sigma_z = sigma_logz * (1 + zred)
    zred_samples = np.random.normal(zred, sigma_z, n_samples)

    # Simplifying assumption that luminosity noise is from  a Gaussian in flux
    # space. In fact it will incorporate some K-correction(z) and K-correction
    # uncertainty.
    flux = 10**(-0.4 * lum_to_mag(logl, zred))
    flux_samples = np.random.normal(flux, sigma_flux, n_samples)

    # now get the luminosty at each z and flux sample
    logl_samples = mag_to_lum(-2.5*np.log10(flux_samples))

    return logl_samples, zred_samples


if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    args.omega = (args.area * arcmin**2).to("steradian").value

    # grid of redshifts
    zgrid = np.linspace(args.zmin, args.zmax, args.nz)

    # grid of log luminosity
    loglgrid = np.linspace(args.loglmin, args.loglmax, args.nl)

    q_true = np.array([0.0, -1.0e-5, 1e-4, 0, 0, 10**(21 / 2.5), -1.7])
    data_samples, veff = make_mock(loglgrid, zgrid, args.omega, q_true,
                                   n_sample=1000)

    lf = EvolvingSchechter()
    lnprobfn = partial(lnlike, data=data_samples, lf=lf, effective_volume=veff)

