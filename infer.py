"""infer.py - likelihood function and posterior sampling routines.
"""

from functools import partial
import numpy as np
from scipy.stats import norm

from lf import EvolvingSchechter

maggies_to_nJy = 3631e9

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

    data : DataSamples instance
        Should have an attribute `all_samples`, structured array with the fields
        'log_samples' and 'zred_samples'

    lf : instance of EvolvingSchecter

    veff : instance of EffectiveVolumeGrid

    Returns
    -------
    lnp : float
        The ln-probability
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
        # This should work when no samples
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
    """Fit model parameters to data using the supplied lnprobfn

    Parameters
    ----------
    params : a Parameters() instantance
        The parameters and their priors

    lnprobfn : callable
        The likelihood function, must take a (ndim,) shaped array of floats
        corresponding to params.free_params

    fitter : string
        Type of fitting to do

    Returns
    -------
    points : ndarray of shape (nsamples, ndim)
        Posterior samples

    log_w : ndarray of shape (nsamples,)
        The ln of the weight for each sample

    log_like : ndarray of shape (nsamples,)
        The ln of the likelihood for each sample

    sampler : object
        The sampler object used in the fit, depends on the fitting method used.
    """

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
        sampler = None

    return points, log_w, log_like, sampler


