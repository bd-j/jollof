#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
from astropy.table import Table

from lf import create_parser
from lf import EvolvingSchechter, sample_twod
from lf import construct_effective_volume
from lf import lum_to_mag, mag_to_lum, arcmin

from priors import Parameters, Uniform


maggies_to_nJy = 3631e9


class DataSamples:
    """Holds posterior samples for logL and zred for all objects being used to
    constrain the LF.  Optionally include flux samples to incorporate k-correction effects or other
    """

    def __init__(self, objects=None, filename=None, ext="SAMPLES", n_samples=1000):

        self.all_samples = None
        self.n_samples = n_samples
        if filename is not None:
            self.all_samples = self.rectify_eazy(filename, ext)
            self.n_samples = len(self.all_samples[0]["zred_samples"])
        if objects is not None:
            self.add_objects(objects)

    def rectify_eazy(self, filename, ext):
        table = Table.read(filename, hdu=ext)
        convert = dict(zred_samples=("z_samples", lambda x: x),
                       logl_samples=("MUV_samples", lambda x: -0.4 * x))

        #table.rename_columns(old, new)
        #table["logl_samples"] = -0.4 * table["logl_samples"]
        #return table.as_array().filled()
        dtype = self.get_dtype(self.n_samples)
        all_samples = np.zeros(len(table), dtype=dtype)
        for n, (o, konvert) in convert.items():
            all_samples[:][n] = konvert(table[o][:, :self.n_samples])
        return all_samples

    def add_objects(self, objects):
        dtype = self.get_dtype(self.n_samples)
        new = np.zeros(len(objects), dtype=dtype)
        for i, obj in enumerate(objects):
            for k in obj.keys():
                new[i][k] = obj[k]
        if self.all_samples is None:
            self.all_samples = new
        else:
            self.all_samples = np.concatenate([self.all_samples, new])

    def get_dtype(self, n_samples):
        truecols = [("logl_true", float), ("zred_true", float)]
        sample_cols = [("logl_samples", float, (n_samples,)),
                       ("zred_samples", float, (n_samples,)),
                       ("sample_weight", float, (n_samples,))]
        flux_cols = [("flux_samples", float, (n_samples,))]
        return np.dtype([("id", int)] + sample_cols + flux_cols + truecols)

    def show(self, n_s=15, ax=None, **plot_kwargs):
        if ax is None:
            fig, ax = pl.subplots()
        else:
            fig = None
        ax.plot(self.all_samples["zred_true"], self.all_samples["logl_true"], "ro",
                zorder=20)
        for d in self.all_samples:
            ax.plot(d["zred_samples"][:n_s], d["logl_samples"][:n_s],
                    marker=".", linestyle="", color='gray')
        if fig is not None:
            fig.savefig("mock_samples.png")
        return fig, ax

    def to_fits(self, fitsfilename):
        samples = fits.BinTableHDU(self.all_samples, name="SAMPLES")
        samples.header["NSAMPL"] = self.n_samples
        hdul = fits.HDUList([fits.PrimaryHDU(),
                             samples])
        hdul.writeto(fitsfilename, overwrite=True)


def make_mock(loglgrid, zgrid, omega,
              q_true, lf=EvolvingSchechter(),
              n_samples=100,
              sigma_logz=0.1, sigma_flux=1/maggies_to_nJy,
              completeness_kwargs={},
              selection_kwargs={}):

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

def transform(qq, lf=None, evolving=False):
    """Transform from sampling parameters to evolving LF parameters
    """
    if evolving:
        q = qq
    else:
        # non-evolving
        q = np.array([qq[0], 0, qq[1], 0, qq[2]])

    if False:
        # transform from knots
        from transforms import knots_to_coeffs
        z_knots = np.array([veff.zgrid.min(),
                            veff.zgrid.mean(),
                            veff.zgrid.max()])
        q = knots_to_coeffs(lf, qq, z_knots=z_knots)

    return q


def lnlike(qq, data=None, veff=None, fast=True,
           lf=EvolvingSchechter(), evolving=False):
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

    q = transform(qq, evolving=evolving)

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
    lnlike[~np.isfinite(lnlike)] = np.nan
    lnp = np.nansum(lnlike) - Neff
    #assert np.isfinite(lnp), debug
    if not np.isfinite(lnp):
        return null

    return lnp


if __name__ == "__main__":

    parser = create_parser()
    parser.add_argument("--fitter", type=str, default="none",
                        choices=["none", "dynesty", "emcee", "brute",
                                 "ultranest", "nautilus"])
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--evolving", type=int, default=0)
    args = parser.parse_args()
    args.omega = (args.area * arcmin**2).to("steradian").value
    sampler_kwargs = dict()

    # grid of redshifts
    zgrid = np.linspace(args.zmin, args.zmax, args.nz)

    # grid of log luminosity
    loglgrid = np.linspace(args.loglmin, args.loglmax, args.nl)

    # ---------------------------
    # --- Set up model and priors
    # ---------------------------
    lf = EvolvingSchechter()
    pdict = dict(phi0=Uniform(mini=-5, maxi=-3),
                 phi1=Uniform(mini=-3, maxi=3),
                 lstar0=Uniform(mini=(19 / 2.5), maxi=(22 / 2.5)),
                 lstar1=Uniform(mini=-3, maxi=3),
                 alpha=Uniform(mini=-2.5, maxi=-1.5))
    if args.evolving:
        param_names = ["phi0", "phi1", "lstar0", "lstar1", "alpha"]
    else:
        param_names = ["phi0", "lstar0", "alpha"]
    params = Parameters(param_names, pdict)

    # -------------------
    # --- Truth ---
    # -------------------
    q_true = np.array([-4, 0, (21 / 2.5), 0, -1.7])
    if args.evolving:
        qq_true = q_true
    else:
        qq_true = np.array([q_true[0], q_true[2], q_true[4]])

    # -----------------------
    # --- build mock data ---
    # -----------------------

    mock, veff = make_mock(loglgrid, zgrid, args.omega,
                           q_true, lf=lf,
                           sigma_logz=0.05,
                           n_samples=args.n_samples)
    print(f"{len(mock.all_samples)} objects drawn from this LF x Veff")
    dN, _ = lf.n_effective(veff)

    fig, ax = pl.subplots()
    ax.imshow(dN, origin="lower", cmap="Blues", alpha=0.5,
              extent=[zgrid.min(), zgrid.max(), loglgrid.min(), loglgrid.max()],
              aspect="auto")
    _, ax = mock.show(ax=ax)
    fig.savefig("mock_samples.png", dpi=300)
    mock.to_fits("mock_data.fits")


    # ---------------
    # --- Fitting ---
    # ---------------
    assert np.isfinite(params.prior_product(qq_true))
    lnprobfn = partial(lnlike, data=mock, lf=lf, veff=veff, evolving=args.evolving)

    if args.evolving:
        def lnprobfn_dict(param_dict):
            qq = np.array([param_dict['phi0'], param_dict['phi1'],
                           param_dict['lstar0'], param_dict['lstar1'],
                           param_dict['alpha']])
            return lnprobfn(qq)
    else:
        def lnprobfn_dict(param_dict):
            qq = np.array([param_dict['phi0'], param_dict['lstar0'], param_dict['alpha']])
            return lnprobfn(qq)


    # -------------------
    # --- Fitting -------
    # -------------------
    if args.fitter == "nautilus":
        from nautilus import Prior, Sampler

        prior = Prior()
        for k in params.param_names:
            prior.add_parameter(k, dist=(pdict[k].params['mini'], pdict[k].params['maxi']))

        sampler = Sampler(prior, lnprobfn_dict, n_live=1000)
        sampler.run(verbose=False)

        import corner
        points, log_w, log_l = sampler.posterior()
        ndim = points.shape[1]
        fig, axes = pl.subplots(ndim, ndim, figsize=(5., 5.))
        fig = corner.corner(points, weights=np.exp(log_w), bins=20, labels=prior.keys,
                            plot_datapoints=False, plot_density=False,
                            fill_contours=True, levels=(0.68, 0.95),
                            range=np.ones(ndim) * 0.999, fig=fig,
                            truths=qq_true, truth_color="red")
        fig.savefig("posteriors-nautilus.png", dpi=300)

    if args.fitter == "ultranest":
        # --- ultranest ---
        import ultranest
        sampler = ultranest.ReactiveNestedSampler(params.free_params, lnprobfn, params.prior_transform)
        result = sampler.run(**sampler_kwargs)
        fig, axes = pl.subplots(ndim, ndim, figsize=(5., 5.))
        fig = corner.corner(points, weights=np.exp(log_w), bins=20, labels=prior.keys,
                            plot_datapoints=False, plot_density=False,
                            fill_contours=True, levels=(0.68, 0.95),
                            range=np.ones(ndim) * 0.999, fig=fig,
                            truths=qq_true, truth_color="red")
        fig.savefig("posteriors-ultranest.png", dpi=300)

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
        fig.savefig("posteriors-dynesty.png")

    if args.fitter == "emcee":
        # --- emcee ---
        assert (not args.evolving)
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
        assert (not args.evolving)
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
