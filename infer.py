#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

from lf import create_parser
from lf import EvolvingSchechter, effective_volume, sample_twod, lum_to_mag

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
    density = lf * veff
    logl_true, zred_true = sample_twod(loglgrid, zgrid, density,
                                       n_sample=n_sample)

    if noisy:
        ll, zz = np.meshgrid(loglgrid, zgrid)
        ff = 10**(-0.4 * lum_to_mag(ll, zz))
        data = []
        for logl, zred in zip(logl_true, zred_true):
            lnp = add_noise(logl, zred, ff, zz,
                            sigma_flux=sigma_flux,
                            sigma_logz=sigma_logz)
            obj = dict(logl=logl, zred=zred, lnp=lnp)
        data.append(obj)
    else:
        data = None

    return logl_true, zred_true, data


def add_noise(logl, zred, ff, zz,
              sigma_flux=1/maggies_to_nJy,
              sigma_logz=0.1):

    mag = lum_to_mag(logl, zred)
    flux = 10**(-0.4 * mag)
    sigma_z = sigma_logz * (1 + zred)
    lnp = -0.5*((flux - ff)/sigma_flux)**2 - 0.5 * ((zred - zz)/sigma_z)**2
    lnp = lnp / lnp.sum()

    return lnp


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

