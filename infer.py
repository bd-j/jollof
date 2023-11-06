#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from functools import partial
import numpy as np
import matplotlib.pyplot as pl

from lf import create_parser
from lf import EvolvingSchechter, sample_twod
from lf import effective_volume, lnlike
from lf import lum_to_mag, mag_to_lum, arcmin

from priors import Parameters, LogUniform, Uniform, Normal, LogNormal
import ultranest


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

    def show(self, n_s=15, **plot_kwargs):
        fig, ax = pl.subplots()
        ax.plot(self.all_samples["zred_true"], self.all_samples["logl_true"], "ro",
                zorder=20)
        for d in self.all_samples:
            ax.plot(d["zred_samples"][:n_s], d["logl_samples"][:n_s],
                    marker=".", linestyle="", color='gray')
        fig.savefig("mock_samples.png")


def make_mock(loglgrid, zgrid, omega,
              q_true,
              n_samples=100,
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
        l_s, z_s = sample_mock_noise(logl, zred,
                                     sigma_logz=sigma_logz, sigma_flux=sigma_flux,
                                     n_samples=n_samples)
        obj = dict(logl_true=logl, zred_true=zred,
                   logl_samples=l_s, zred_samples=z_s)
        data.append(obj)
    alldata = DataSamples(data, n_samples=1000)

    return alldata, veff


def sample_mock_noise(logl, zred, n_samples=1000,
                      sigma_flux=1/maggies_to_nJy,  # 1nJy limit
                      sigma_logz=0.1):
    if n_samples == 1:
        return np.array([logl]), np.array([zred])
    # sample the p(z) distribution
    dz_1pz = np.random.normal(0, sigma_logz, n_samples)
    zred_samples = zred + dz_1pz * (1 + zred)

    # Simplifying assumption that luminosity noise is from  a Gaussian in flux
    # space. In fact it will incorporate some K-correction(z) and K-correction
    # uncertainty. Also the luminosity should never go below zero
    flux = 10**(-0.4 * lum_to_mag(logl, zred))
    flux_samples = np.random.normal(flux, sigma_flux, n_samples)
    epsilon = sigma_flux / 10
    flux_samples = np.clip(flux_samples, epsilon, np.inf)

    # now get the luminosty at each z and flux sample
    logl_samples = mag_to_lum(-2.5*np.log10(flux_samples), zred_samples)

    return logl_samples, zred_samples



if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    args.omega = (args.area * arcmin**2).to("steradian").value
    sampler_kwargs = dict()

    # grid of redshifts
    zgrid = np.linspace(args.zmin, args.zmax, args.nz)

    # grid of log luminosity
    loglgrid = np.linspace(args.loglmin, args.loglmax, args.nl)

    q_true = np.array([0.0, -1.0e-5, 1e-4, 0, 0, 10**(21 / 2.5), -1.7])
    data_samples, veff = make_mock(loglgrid, zgrid, args.omega, q_true,
                                   sigma_logz=0.01,
                                   n_samples=1)

    data_samples.show()

    lf = EvolvingSchechter()
    lnprobfn = partial(lnlike, data=data_samples, lf=lf, effective_volume=veff)

    priors = dict(phi2=Normal(mean=0, sigma=1e-6),
                  phi1=Normal(mean=0, sigma=2e-5),
                  phi0=LogUniform(mini=1e-5, maxi=1e-3),
                  lstar2=Normal(mean=0, sigma=1e-2),
                  lstar1=Normal(mean=0, sigma=1e-1),
                  lstar0=LogUniform(mini=10**(19/2.5), maxi=10**(22/2.5)),
                  alpha=Uniform(mini=-2.5, maxi=-1.5))
    param_names = ["phi2", "phi1", "phi0", "lstar2", "lstar1", "lstar0", "alpha"]
    params = Parameters(param_names, priors)
    assert np.isfinite(params.prior_product(q_true))


    sampler = ultranest.ReactiveNestedSampler(params.free_params, lnprobfn, params.prior_transform)


    sampler.run(**sampler_kwargs)
    sys.exit()