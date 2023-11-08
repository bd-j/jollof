#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
from astropy.table import Table

from infer import lnlike, DataSamples

from lf import create_parser
from lf import EvolvingSchechter, construct_effective_volume
from lf import arcmin

from priors import Parameters, Uniform


if __name__ == "__main__":

    parser = create_parser()
    parser.add_argument("--fitter", type=str, default="none",
                        choices=["none", "dynesty", "emcee", "brute",
                                 "ultranest", "nautilus"])
    parser.add_argument("--n_samples", type=int, default=1)
    args = parser.parse_args()
    args.omega = (args.area * arcmin**2).to("steradian").value
    sampler_kwargs = dict()

    # grid of redshifts
    zgrid = np.linspace(args.zmin, args.zmax, args.nz)

    # grid of log luminosity
    loglgrid = np.linspace(args.loglmin, args.loglmax, args.nl)

    # -------------------
    # --- Initial ---
    # -------------------
    q_true = np.array([0.0, 0.0, 1e-4, 0, 0.0, 10**(21 / 2.5), -1.7])
    #z_knots = np.array([zgrid.min(), zgrid.mean(), zgrid.max()])
    #qq_true = lf.coeffs_to_knots(q_true, z_knots)
    #assert np.allclose(lf.knots_to_coeffs(qq, z_knots), q_true)
    qq_true = np.array([np.log10(q_true[2]), np.log10(q_true[5]), q_true[6]])

    # -----------------------
    # --- read data ---
    # -----------------------
    veff = construct_effective_volume(loglgrid, zgrid, omega=args.omega,
                                      as_interpolator=True)
    jof = DataSamples(filename="data/samples.v094.11072023.fits",
                      ext="ZSAMP",
                      n_samples=args.n_samples)
    jof.show(filename="jof_samples.png")

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

    lnprobfn = partial(lnlike, data=jof, lf=lf, veff=veff)


    # -------------------
    # --- Fitting -------
    # -------------------
    if args.fitter == "nautilus":
        from nautilus import Prior, Sampler

        prior = Prior()
        for k, p in priors.items():
            prior.add_parameter(k, dist=(p.params['mini'], p.params['maxi']))
        def lnprobfn_dict(param_dict):
            qq = np.array(np.array([param_dict['phi0'], param_dict['lstar0'], param_dict['alpha']]))
            return lnprobfn(qq)

        sampler = Sampler(prior, lnprobfn_dict, n_live=1000)
        sampler.run(verbose=False)

        import corner
        points, log_w, log_l = sampler.posterior()
        ndim = points.shape[1]
        fig, axes = pl.subplots(ndim, ndim, figsize=(3.5, 3.5))
        fig = corner.corner(points, weights=np.exp(log_w), bins=20, labels=prior.keys,
                            plot_datapoints=False, plot_density=False,
                            fill_contours=True, levels=(0.68, 0.95),
                            range=np.ones(ndim) * 0.999, fig=fig,
                            truths=qq_true, truth_color="red")
        fig.savefig("jof-posteriors-nautilus.png")

    if args.fitter == "ultranest":
        # --- ultranest ---
        import ultranest
        sampler = ultranest.ReactiveNestedSampler(params.free_params, lnprobfn, params.prior_transform)
        result = sampler.run(**sampler_kwargs)
        from ultranest.plot import cornerplot
        fig, axes = cornerplot(result)

    if args.fitter == "dynesty":
        # --- Dynesty ---
        import dynesty
        dsampler = dynesty.DynamicNestedSampler(lnprobfn, params.prior_transform,
                                                len(params.free_params),
                                                nlive=1000,
                                                bound='multi', sample="unif",
                                                walks=48)
        dsampler.run_nested(n_effective=1000, dlogz_init=0.05)
        from dynesty import plotting as dyplot
        fig, axes = dyplot.cornerplot(dsampler.results, labels=[r"$\phi_*$", r"L$_*$", r"$\alpha$"], truths=qq_true)
        fig.savefig("jof-posteriors-dynesty.png")

    if args.fitter == "emcee":
        # --- emcee ---
        def lnposterior(qq, params=None, data=None, lf=None, veff=None):
            # need to include the prior for emcee
            lnp = params.prior_product(qq)
            lnl = lnlike(qq, data=data, lf=lf, veff=veff)
            return lnp + lnl
        lnprobfn = partial(lnposterior, params=params, data=jof, lf=lf, veff=veff)
        import emcee

        nwalkers, ndim, niter = 32, len(qq_true), 512
        initial = np.array([params.prior_transform(u)
                            for u in np.random.uniform(0, 1, (nwalkers, ndim))])

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobfn)
        sampler.run_mcmc(initial, niter, progress=True)

    if args.fitter == "brute":
        # -- Brute Force on a grid ---
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
