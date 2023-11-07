#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from functools import partial
import numpy as np
import matplotlib.pyplot as pl

from lf import create_parser
from lf import EvolvingSchechter, sample_twod
from lf import construct_effective_volume
from lf import lum_to_mag, mag_to_lum, arcmin

from priors import Parameters, LogUniform, Uniform, Normal, LogNormal

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
    veff = construct_effective_volume(loglgrid, zgrid, omega,
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
    alldata = DataSamples(data, n_samples=n_samples)

    return alldata, veff


def sample_mock_noise(logl, zred, n_samples=1,
                      sigma_flux=1/maggies_to_nJy,  # 1nJy limit
                      sigma_logz=0.1):
    if n_samples == 1:
        # noiseless - each object is represented by a single delta-fn in L-z space
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


# ------------------------
# Log likelihood L(data|q)
# ------------------------
def lnlike(qq, data=None, veff=None, lf=EvolvingSchechter(), fast=True):
    """
    Parameters
    ----------
    qq : ndarray
        LF parameters, in terms of knots of the evolving LF

    data : list of dicts
        One dict for each object.  The dictionary should have the keys
        'log_samples' and 'zred_samples'

    lf : instance of EvolvingSchecter

    veff : instance of EffectiveVolumeGrid
    """
    null = -np.inf
    # transform from knot values to evolutionary params
    #z_knots = np.array([effective_volume.zgrid.min(),
    #                    effective_volume.zgrid.mean(),
    #                    effective_volume.zgrid.max()])
    #q = lf.knots_to_coeffs(qq, z_knots=z_knots)

    q = np.array([0, 0, 10**qq[0], 0, 0, 10**qq[1], qq[2]])

    debug = f"q=np.array([{', '.join([str(qi) for qi in q])}])"
    debug += f"\nqq=np.array([{', '.join([str(qi) for qi in qq])}])"
    lf.set_parameters(q)
    dN, dV = lf.n_effective(veff)
    Neff = np.nansum(dN)
    if Neff <= 0:
        # this can happen in the evolving LF case for pathological parameters...
        return null

    if fast:
        # compute likelihood of all objects, equal number of samples each
        # not actually faster!
        l_s, z_s = data.all_samples["logl_samples"], data.all_samples["zred_samples"]
        n_g, n_s = l_s.shape
        l_s, z_s = l_s.flatten(), z_s.flatten()
        p_lf = lf.evaluate(10**l_s, z_s, grid=False, in_dlogl=True)  # ~40% of time
        v_eff_value = veff(np.array([l_s, z_s]).T)   # ~40% of time
        like = (p_lf * v_eff_value).reshape(n_g, n_s)
        lnlike = np.log(np.nansum(like, axis=-1)) - np.log(n_s)

    else:
        # compute likelihood of each object, allowing for ragged samples
        lnlike = np.zeros(len(data.all_samples))
        for i, d in enumerate(data.all_samples):
            l_s, z_s = d["logl_samples"], d["zred_samples"]
            # TODO: in_dlogl = True/False?
            p_lf = lf.evaluate(10**l_s, z_s, grid=False, in_dlogl=True)
            # case where some or all samples are outside the grid is handled by
            # giving them zero Veff (but they still contribute to 1/N_samples
            # weighting)
            # TODO: store the data in this format so we don't have to create arrays every time.
            v_eff_value = veff(np.array([l_s, z_s]).T)
            like = np.nansum(p_lf * v_eff_value) / len(l_s)
            lnlike[i] = np.log(like)

    # Hacks for places where likelihood of all data is ~ 0
    lnp = np.nansum(lnlike) - Neff
    #assert np.isfinite(lnp), debug
    if not np.isfinite(lnp):
        return null

    return lnp


if __name__ == "__main__":

    parser = create_parser()
    parser.add_argument("--fitter", type=str, default="none",
                        choices=["none", "dynesty", "emcee", "brute", "ultranest"])
    args = parser.parse_args()
    args.omega = (args.area * arcmin**2).to("steradian").value
    sampler_kwargs = dict()

    # grid of redshifts
    zgrid = np.linspace(args.zmin, args.zmax, args.nz)

    # grid of log luminosity
    loglgrid = np.linspace(args.loglmin, args.loglmax, args.nl)

    # -------------------
    # --- Truth ---
    # -------------------
    q_true = np.array([0.0, 0.0, 1e-4, 0, 0.0, 10**(21 / 2.5), -1.7])
    #z_knots = np.array([zgrid.min(), zgrid.mean(), zgrid.max()])
    #qq_true = lf.coeffs_to_knots(q_true, z_knots)
    #assert np.allclose(lf.knots_to_coeffs(qq, z_knots), q_true)
    qq_true = np.array([np.log10(q_true[2]), np.log10(q_true[5]), q_true[6]])

    # -----------------------
    # --- build mock data ---
    # -----------------------

    mock, veff = make_mock(loglgrid, zgrid, args.omega, q_true,
                           sigma_logz=0.01,
                           n_samples=1)
    print(f"{len(mock.all_samples)} objects drawn from this LF x Veff")
    mock.show()

    # ---------------------------
    # --- Set up model and priors
    # ---------------------------

    lf = EvolvingSchechter()
    priors = dict(#phi2=LogUniform(mini=1e-5, maxi=1e-3),
                  #phi1=LogUniform(mini=1e-5, maxi=1e-3),
                  phi0=Uniform(mini=-5, maxi=-3),
                  #lstar2=LogUniform(mini=10**(19/2.5), maxi=10**(22/2.5)),
                  #lstar1=LogUniform(mini=10**(19/2.5), maxi=10**(22/2.5)),
                  lstar0=Uniform(mini=(19 / 2.5), maxi=(22 / 2.5)),
                  alpha=Uniform(mini=-2.5, maxi=-1.5))
    #param_names = ["phi2", "phi1", "phi0", "lstar2", "lstar1", "lstar0", "alpha"]
    param_names = ["phi0", "lstar0", "alpha"]
    params = Parameters(param_names, priors)
    assert np.isfinite(params.prior_product(qq_true))

    lnprobfn = partial(lnlike, data=mock, lf=lf, veff=veff)


    # -------------------
    # --- Fitting -------
    # -------------------

    if args.fitter == "ultranest":
        # --- ultranest ---
        lnprobfn = partial(lnlike, data=mock, lf=lf, veff=veff)
        import ultranest
        sampler = ultranest.ReactiveNestedSampler(params.free_params, lnprobfn, params.prior_transform)
        sampler.run(**sampler_kwargs)

    if args.fitter == "dynesty":
        # --- Dynesty ---
        lnprobfn = partial(lnlike, data=mock, lf=lf, veff=veff)
        import dynesty
        dsampler = dynesty.DynamicNestedSampler(lnprobfn, params.prior_transform,
                                                len(params.free_params),
                                                nlive=1000,
                                                bound='multi', sample="unif",
                                                walks=48)
        dsampler.run_nested(n_effective=1000, dlogz_init=0.05)
        from dynesty import plotting as dyplot
        fig, axes = dyplot.cornerplot(dsampler.results, labels=[r"$\phi_*$", r"L$_*$", r"$\alpha$"], truths=qq_true)
        fig.savefig("dynesty_posteriors.png")

    if args.fitter == "emcee":
        # --- emcee ---
        def lnposterior(qq, params=None, data=None, lf=None, veff=None):
            # need to include the prior for emcee
            lnp = params.prior_product(qq)
            lnl = lnlike(qq, data=data, lf=lf, veff=veff)
            return lnp + lnl
        lnprobfn = partial(lnposterior, params=params, data=mock, lf=lf, veff=veff)
        import emcee

        nwalkers, ndim, niter = 32, len(qq_true), 512
        initial = np.array([params.prior_transform(u)
                            for u in np.random.uniform(0, 1, (nwalkers, ndim))])

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobfn)
        sampler.run_mcmc(initial, niter, progress=True)

    if args.fitter == "brute":
        # -- Brute Force on a grid ---
        lnprobfn = partial(lnlike, data=mock, lf=lf, veff=veff)
        from itertools import product
        phi_grid = 10**np.linspace(-5, -3, 30)
        lstar_grid = 10**np.linspace(19/2.5, 22/2.5, 30)
        alpha_grid = np.linspace(-2.5, -1.5, 30)
        qqs = np.array(list(product(phi_grid, lstar_grid, alpha_grid)))
        lnp = np.zeros(len(qqs))
        for i, qq in enumerate(qqs):
            lnp[i] = lnprobfn(qq)

        p = np.exp(lnp - lnp.max())
        p[~np.isfinite(p)] = 0
        fig, ax = pl.subplots()
        ax.clear()
        ax.hexbin(np.log10(qqs[:, 0]), -2.5 * np.log10(qqs[:, 1]), C=p, gridsize=25)
        ax.set_xlabel(r"$\phi_{*}$")
        #ax.set_xscale("log")
        ax.set_ylabel(r"M$_{*,\rm UV}$")
        ax.plot(np.log10(qq_true[0]), -2.5 * np.log10(qq_true[1]), "ro")
        fig, ax = pl.subplots()
        ax.clear()
        ax.hexbin(qqs[:, 2], -2.5 * np.log10(qqs[:, 1]), C=p, gridsize=25)
        ax.set_xlabel(r"$\alpha$")
        #ax.set_xscale("log")
        ax.set_ylabel(r"M$_{*,\rm UV}$")
        ax.plot(qq_true[2], -2.5 * np.log10(qq_true[1]), "ro")
