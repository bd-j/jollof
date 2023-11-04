#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

from lf import create_parser
from lf import EvolvingSchechter, effective_volume, sample_twod
from lf import lum_to_mag, mag_to_lum

maggies_to_nJy = 3631e9


def make_mock(loglgrid, zgrid, omega,
              q_true,
              n_sample=100,
              noisy=False,
              sigma_logz=0.1, sigma_flux=1/maggies_to_nJy,
              completeness_kwargs={},
              selection_kwargs={}):
    lgrid = 10**loglgrid
    s = EvolvingSchechter()
    s.set_parameters(q_true)
    lf = s.evaluate(lgrid, zgrid)
    veff = effective_volume(loglgrid, zgrid, omega,
                            completeness_kwargs=completeness_kwargs,
                            selection_kwargs=selection_kwargs)

    dN_dz_dlogl = lf * veff
    dN = dN_dz_dlogl * np.gradient(zgrid) * np.gradient(loglgrid)[:, None]
    N_bar = dN.sum()
    N = np.random.poisson(N_bar, size=1)[0]

    logl_true, zred_true = sample_twod(loglgrid, zgrid, dN,
                                       n_sample=N)

    if noisy:
        data = []
        for logl, zred in zip(logl_true, zred_true):
            l_s, z_s = sample_mock_noise(logl, zred)
            obj = dict(logl_true=logl, zred_true=zred,
                       logl_samples=l_s, zred_samples=z_s)
        data.append(obj)
    else:
        data = None

    return logl_true, zred_true, data


def sample_mock_noise(logl, zred, n_samples=1000,
                      sigma_flux=1/maggies_to_nJy,  # 1nJy limit
                      sigma_logz=0.1):
    # sample the p(z) distribution
    sigma_z = sigma_logz * (1 + zred)
    zred_samples =  np.random.normal(zred, sigma_z, n_samples)

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

    from astropy import units as u
    area = 2 * u.arcmin * 5*u.arcmin
    area = area.to("steradian").value

    # grid of redshifts
    zgrid = np.linspace(args.zmin, args.zmax, args.nz)

    # grid of log luminosity
    loglgrid = np.linspace(args.loglmin, args.loglmax, args.nl)

    q_true = np.array([0.0, -1.0e-4, 1e-3, 0, 0, 10**(21 / 2.5), -1.5])
    logl, zred, data = make_mock(loglgrid, zgrid, area, q_true,
                                 n_sample=100,
                                 noisy=False)

