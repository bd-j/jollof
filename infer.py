#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
from astropy.table import Table
from scipy.stats import norm

from lf import create_parser
from lf import EvolvingSchechter, sample_twod
from lf import construct_effective_volume
from lf import lum_to_mag, mag_to_lum, arcmin

from priors import Parameters, Uniform, Normal


maggies_to_nJy = 3631e9


class DataSamples:
    """Holds posterior samples for logL and zred for all objects being used to
    constrain the LF.  Optionally include flux samples to incorporate k-correction effects or other
    """

    def __init__(self, objects=None, filename=None, ext="SAMPLES", n_samples=1000, replicate=0):

        self.all_samples = None
        self.n_samples = n_samples
        if filename is not None:
            self.all_samples = self.rectify_eazy(filename, ext, replicate=replicate)
            self.n_samples = len(self.all_samples[0]["zred_samples"])
        if objects is not None:
            self.add_objects(objects)

    def rectify_eazy(self, filename, ext, replicate=0):
        table = Table.read(filename, hdu=ext)
        convert = dict(zred_samples=("z_samples", lambda x: x),
                       logl_samples=("MUV_samples", lambda x: -0.4 * x))

        dtype = self.get_dtype(self.n_samples)
        all_samples = np.zeros(len(table), dtype=dtype)
        for n, (o, konvert) in convert.items():
            all_samples[:][n] = konvert(table[o][:, :self.n_samples])

        #Add duplicate objects (for testing purposes)
        if(replicate>0):
            print(f'Replicating the sample {replicate} times.')
            print(f'all_samples.shape {all_samples.shape} all_samples.keys() {all_samples[0].dtype}')
            new_samples = all_samples.copy()
            for i in range(replicate):
                new_samples = np.append(new_samples,all_samples)
            all_samples = new_samples
            print(f'all_samples.shape {all_samples.shape} all_samples.keys() {all_samples[0].dtype}')

        return all_samples

    def to_eazy(self):
        convert = dict(z_samples=("zred_samples", lambda x: x),
                       MUV_samples=("logl_samples", lambda x: -2.5 * x))
        n_samples = self.n_samples
        dtype = np.dtype([("MUV_samples", float, (n_samples,)),
                          ("z_samples", float, (n_samples,))])
        arr = np.zeros(len(self.all_samples), dtype=dtype)
        for n, (o, konvert) in convert.items():
            arr[:][n] = konvert(self.all_samples[o][:, :self.n_samples])
        return arr

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
        arr = self.to_eazy()
        samples = fits.BinTableHDU(arr, name="SAMPLES")
        samples.header["NSAMPL"] = self.n_samples
        hdul = fits.HDUList([fits.PrimaryHDU(),
                             samples])
        hdul.writeto(fitsfilename, overwrite=True)


def make_mock(loglgrid, zgrid, omega,
              q_true, lf=EvolvingSchechter(),
              n_samples=100,
              sigma_logz=0.1, sigma_flux=1/maggies_to_nJy,
              completeness_kwargs={},
              selection_kwargs={},
              fake_flag=True,
              seed=None):
    """Draw a set of mock galaxies from a LF x Veff, add noise to them and draew
    samples from that noise
    """

    #np.random.seed(seed)

    lf.set_parameters(q_true)
    veff = construct_effective_volume(loglgrid, zgrid, omega,
                                      completeness_kwargs=completeness_kwargs,
                                      selection_kwargs=selection_kwargs,
                                      fake_flag=fake_flag,
                                      as_interpolator=True)

    dN, dV = lf.n_effective(veff)
    N_bar = dN.sum()
    N = np.random.poisson(N_bar, size=1)[0]
    logl_true, zred_true = sample_twod(loglgrid, zgrid, dN,
                                       n_sample=N)

    data = []
    for logl, zred in zip(logl_true, zred_true):
        l_s, z_s = sample_mock_noise(logl, zred,
                                     sigma_logz=sigma_logz,
                                     sigma_flux=sigma_flux,
                                     n_samples=n_samples)
        obj = dict(logl_true=logl,
                   zred_true=zred,
                   logl_samples=l_s,
                   zred_samples=z_s)
        data.append(obj)
    alldata = DataSamples(data, n_samples=n_samples)
    alldata.seed = seed

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
        if(evolving==2):
            q = np.array([qq[0], qq[1], qq[2], 0, qq[3]])
        else:
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


# ------------------------
# Fitting
# -----------------------
def fit(params, lnprobfn, verbose=False,
        fitter="nautilus", sampler_kwargs=dict()):

    if fitter == "nautilus":
        from nautilus import Prior, Sampler

        # we have to use the nautilus prior objects
        prior = Prior()
        for k in params.param_names:
            pr = params.priors[k]
            if pr.kind == "Normal":
                prior.add_parameter(k, dist=norm(pr.params['mean'], pr.params['sigma']))
            else:
                prior.add_parameter(k, dist=(pr.params['mini'], pr.params['maxi']))
        sampler = Sampler(prior, lnprobfn, n_live=1000)
        sampler.run(verbose=verbose)

        points, log_w, log_like = sampler.posterior()

    if fitter == "ultranest":
        # --- ultranest ---
        import ultranest
        sampler = ultranest.ReactiveNestedSampler(params.free_params, lnprobfn, params.prior_transform)
        result = sampler.run(**sampler_kwargs)

        points = np.array(result['weighted_samples']['points'])
        log_w = np.log(np.array(result['weighted_samples']['weights']))
        log_like = np.array(result['weighted_samples']['logl'])

    if fitter == "dynesty":
        # --- Dynesty ---
        import dynesty
        sampler = dynesty.DynamicNestedSampler(lnprobfn, params.prior_transform,
                                               len(params.free_params),
                                               nlive=1000,
                                               bound='multi',
                                               sample="unif",
                                               walks=48)
        sampler.run_nested(n_effective=1000, dlogz_init=0.05)

        points = sampler.results["samples"]
        log_w = sampler.results["logwt"]
        log_like = sampler.results["logl"]

    if fitter == "emcee":
        raise(NotImplementedError)
        assert (not args.evolving)
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

        points = sampler.flatchain
        log_w = None
        log_like = sampler.flatlnprobability

    if fitter == "brute":
        raise(NotImplementedError)
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

        points = qq
        log_w = None
        log_like = lnp

    return points, log_w, log_like, sampler


if __name__ == "__main__":

    parser = create_parser()
    parser.add_argument("--fitter", type=str, default="none",
                        choices=["none", "dynesty", "emcee", "brute",
                                 "ultranest", "nautilus"])
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--evolving", type=int, default=0)
    parser.add_argument("--sample_output", type=str, default='samples.fits')

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
                 alpha=Normal(mean=-2, sigma=0.05))
    if args.evolving:
        if(args.evolving==2):
            param_names = ["phi0", "phi1", "lstar0", "alpha"]
        else:
            param_names = ["phi0", "phi1", "lstar0", "lstar1", "alpha"]
    else:
        param_names = ["phi0", "lstar0", "alpha"]
    params = Parameters(param_names, pdict)

    # -------------------
    # --- Truth ---
    # -------------------
    q_true = np.array([-4, 0, (21 / 2.5), 0, -2.0])
    if args.evolving:
        if(args.evolving==2):
            qq_true = np.array([q_true[0], q_true[1], q_true[2], q_true[4]])
        else:
            qq_true = q_true
    else:
        qq_true = np.array([q_true[0], q_true[2], q_true[4]])

    # -----------------------
    # --- build mock data ---
    # -----------------------
    mock, veff = make_mock(loglgrid, zgrid, args.omega,
                           q_true, lf=lf,
                           sigma_logz=0.05,
                           sigma_flux=0.1/maggies_to_nJy,
                           n_samples=args.n_samples,
                           fake_flag=args.fake_flag)
    print(f"{len(mock.all_samples)} objects drawn from this LF x Veff")
    dN, _ = lf.n_effective(veff)

    fig, ax = pl.subplots()
    ax.imshow(dN, origin="lower", cmap="Blues", alpha=0.5,
              extent=[zgrid.min(), zgrid.max(), loglgrid.min(), loglgrid.max()],
              aspect="auto")
    _, ax = mock.show(ax=ax)
    fig.savefig("mock_samples.png", dpi=300)
    mock.to_fits("mock_data.fits")
    veff.to_fits("mock_veff.fits")

    #sys.exit()

    # ----------------
    # --- lnprobfn ---
    # ----------------
    assert np.isfinite(params.prior_product(qq_true))
    lnprobfn = partial(lnlike, data=mock, lf=lf, veff=veff, evolving=args.evolving)

    if args.evolving:
        if(args.evolving==2):
            def lnprobfn_dict(param_dict):
                qq = np.array([param_dict['phi0'], param_dict['phi1'],
                           param_dict['lstar0'],
                           param_dict['alpha']])
                return lnprobfn(qq)
        else:
            def lnprobfn_dict(param_dict):
                qq = np.array([param_dict['phi0'], param_dict['phi1'],
                           param_dict['lstar0'], param_dict['lstar1'],
                           param_dict['alpha']])
                return lnprobfn(qq)
    else:
        def lnprobfn_dict(param_dict):
            qq = np.array([param_dict['phi0'], param_dict['lstar0'], param_dict['alpha']])
            return lnprobfn(qq)

    if args.fitter == "nautilus":
        lnprob = lnprobfn_dict
    else:
        lnprob = lnprobfn
    points, log_w, log_like, sampler = fit(params, lnprob, fitter=args.fitter)

    # ---------------
    # --- Plotting ---
    # ---------------

    import corner
    mle = points[np.argmax(log_like)]
    ndim = points.shape[1]
    pmean = np.zeros(ndim)
    for i in range(ndim):
        pmean[i] = np.sum(np.exp(log_w)*points[:,i])/np.sum(np.exp(log_w))
        print(f'Mean {pmean[i]}')

    print(f'Shape of points {points.shape}')
    fig, axes = pl.subplots(ndim, ndim, figsize=(6., 6.))
    fig = corner.corner(points, weights=np.exp(log_w), bins=20, labels=params.free_params,
                        plot_datapoints=False, plot_density=False,
                        fill_contours=True, levels=(0.68, 0.95),
                        range=np.ones(ndim) * 0.999, fig=fig,
                        truths=qq_true, truth_color="red")

    fig.suptitle(f"Mock: {qq_true}")
    fig.savefig(f"posteriors-{args.fitter}.png", dpi=300)
    print(f"MAP={points[np.argmax(log_like)]}")


    # ---------------
    # Save samples to a fits file
    # ---------------
    sample_table = Table()
    sample_table['phistar'] = points[:,0]
    sample_table['mstar']   = -2.5*points[:,1]
    sample_table['alpha']   = points[:,2]
    sample_table['loglike'] = log_like
    sample_table['logw']    = log_w
    sample_table.write(args.sample_output,format='fits',overwrite=True)
