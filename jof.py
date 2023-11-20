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
from lf import plot_selection, plot_detection, CompletenessGrid
from lf import plot_veff

from priors import Parameters, Uniform, Normal
from scipy.stats import norm, gamma



if __name__ == "__main__":

    parser = create_parser()
    parser.add_argument("--fitter", type=str,
                        default="none",
                        choices=["none", "dynesty", "emcee", "brute",
                                 "ultranest", "nautilus"])
    parser.add_argument("--n_samples", type=int,
                        default=1)
    parser.add_argument("--evolving", type=int,
                        default=0)
    parser.add_argument("--jof_datafile", type=str,
                        default="data/samples.v094.11072023.fits")
    parser.add_argument("--detection", type=str,
                        default=None)
    parser.add_argument("--selection", type=str,
                        default=None)
    parser.add_argument("--sample_output", type=str,
                        default='samples.fits')
    parser.add_argument("--replicate", type=int,
                        default=0)
    #create parser
    args = parser.parse_args()

    #print important arguments to the
    #screen
    if(args.verbose):
        print(f'Number of samples {args.n_samples}')
        print(f'Minimum redshift {args.zmin}')
        print(f'Maximum redshift {args.zmax}')
        print(f'Minimum Muv {args.loglmin*-2.5}')
        print(f'Maximum Muv {args.loglmax*-2.5}')
        print(f'Replicate input objects? {args.replicate}')



    #compute area
    args.omega = (args.area * arcmin**2).to("steradian").value
    sampler_kwargs = dict()

    # grid of redshifts
    zgrid = np.linspace(args.zmin, args.zmax, args.nz)

    # grid of log luminosity
    loglgrid = np.linspace(args.loglmin, args.loglmax, args.nl)

    # -----------------------
    # --- read data ---
    # -----------------------
    jof = DataSamples(filename=args.jof_datafile,
                      ext="ZSAMP",
                      n_samples=args.n_samples, replicate=args.replicate)
    sfig, sax = jof.show(n_s=jof.n_samples, filename="jof_samples.png")
    sax.set_ylim(0.95*jof.all_samples["logl_samples"].min(), 1.1*jof.all_samples["logl_samples"].max())
    sax.set_xlim(0.95*jof.all_samples["zred_samples"].min(), 1.1*jof.all_samples["zred_samples"].max())
    sfig.savefig("jof_samples.png", dpi=300)
    pl.close(sfig)


    # ------------------------------------
    # ---- read detection completeness ---
    # ------------------------------------
    if(args.detection is not None):
        mab_det_grid  = np.asarray(fits.getdata(args.detection,'MAB'),dtype=np.float32)
        lrh_det_grid  = np.asarray(fits.getdata(args.detection,'LOGRHALF'),dtype=np.float32)
        comp_det_grid = np.asarray(fits.getdata(args.detection,'DET_COMP'),dtype=np.float32)



        #get an interpolator for the detection completeness
        comp_det = CompletenessGrid(comp_det_grid,mab_det_grid,lrh_det_grid)

        #save a figure
        plot_detection(mab_det_grid,lrh_det_grid, comp_det)

    # ------------------------------------
    # ---- read selection completeness ---
    # ------------------------------------
    if(args.selection is not None):
        muv_sel_grid  = np.asarray(fits.getdata(args.selection,'MUV'),dtype=np.float32)
        z_sel_grid    = np.asarray(fits.getdata(args.selection,'Z'),dtype=np.float32)
        comp_sel_grid = np.asarray(fits.getdata(args.selection,'SEL_COMP'),dtype=np.float32)

        #get an interpolator for the selection completeness
        comp_sel = CompletenessGrid(comp_sel_grid,z_sel_grid,muv_sel_grid)

        #save a figure
        plot_selection(z_sel_grid,muv_sel_grid,comp_sel)

    # ------------------------------------
    # ---- compute effective volume    ---
    # ------------------------------------
    veff = construct_effective_volume(loglgrid, zgrid, omega=args.omega,\
                                      as_interpolator=True, fake_flag=False,\
                                      muv_min=mab_det_grid.min(),muv_max=mab_det_grid.max(),\
                                      comp_det=comp_det, comp_sel=comp_sel)
    #save a figure
    plot_veff(loglgrid,zgrid,veff)

    # ---------------------------
    # --- Set up model and priors
    # ---------------------------
    lf = EvolvingSchechter()
    #pdict = dict(phi0=Uniform(mini=-5, maxi=-3),
                 #lstar0=Uniform(mini=(17 / 2.5), maxi=(22 / 2.5)),
    pdict = dict(  
                 #phi0=Uniform(mini=-6, maxi=-2),
                 phi0=Uniform(mini=-8, maxi=-2),\
                 phi1=Uniform(mini=-3, maxi=3),\
                 lstar0=Uniform(mini=(17 / 2.5), maxi=(24 / 2.5)),\
                 lstar1=Uniform(mini=-3, maxi=3),\
                 alpha=Normal(mean=-2.0, sigma=0.1))
#                 alpha=Uniform(mini=-2.5, maxi=-1.5))

    if args.evolving:
        param_names = ["phi0", "phi1", "lstar0", "lstar1", "alpha"]
    else:
        param_names = ["phi0", "lstar0", "alpha"]
    params = Parameters(param_names, pdict)

    # -------------------
    # --- Initial ---
    # -------------------
    q_true = np.array([-4, 0, (21 / 2.5), 0, -1.7])
    if args.evolving:
        qq_true = q_true
    else:
        qq_true = np.array([q_true[0], q_true[2], q_true[4]])

    # -------------------
    # --- lnprobfn -------
    # -------------------
    assert np.isfinite(params.prior_product(qq_true))
    lnprobfn = partial(lnlike, data=jof, lf=lf, veff=veff, evolving=args.evolving)

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
    # --- fitting -------
    # -------------------
    if args.fitter == "nautilus":
        from nautilus import Prior, Sampler

        prior = Prior()
        for k in params.param_names:
            if((k=='phi0')|(k=='lstar0')):
                prior.add_parameter(k, dist=(pdict[k].params['mini'], pdict[k].params['maxi']))
            else:
                prior.add_parameter(k, dist=norm(pdict[k].params['mean'], pdict[k].params['sigma']))
#                prior.add_parameter(k, dist=(-3, -1))

        #sampler = Sampler(prior, lnprobfn_dict, n_live=1000)
        sampler = Sampler(prior, lnprobfn_dict, n_live=1000)
        sampler.run(verbose=args.verbose)

        points, log_w, log_like = sampler.posterior()

    if args.fitter == "ultranest":
        # --- ultranest ---
        import ultranest
        sampler = ultranest.ReactiveNestedSampler(params.free_params, lnprobfn, params.prior_transform)
        result = sampler.run(**sampler_kwargs)

        points = np.array(result['weighted_samples']['points'])
        log_w = np.log(np.array(result['weighted_samples']['weights']))
        log_like = np.array(result['weighted_samples']['logl'])

    if args.fitter == "dynesty":
        # --- Dynesty ---
        import dynesty
        dsampler = dynesty.DynamicNestedSampler(lnprobfn, params.prior_transform,
                                                len(params.free_params),
                                                nlive=1000,
                                                bound='multi', sample="unif",
                                                walks=48)
        dsampler.run_nested(n_effective=1000, dlogz_init=0.05)

        points = dsampler.results["samples"]
        log_w = dsampler.results["logwt"]
        log_like = dsampler.results["logl"]

    if args.fitter == "emcee":
        # --- emcee ---
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

        points = qq
        log_w = None
        log_like = lnp

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
                        truths=pmean, truth_color="red")
#                        truths=mle, truth_color="red")

    fig.suptitle(args.jof_datafile)
    fig.savefig(f"jof-posteriors-{args.fitter}.png", dpi=300)
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


    # - luminosity density -
    from infer import transform
    from lf import Maggie_to_cgs
    qq = np.array([transform(p, evolving=args.evolving) for p in points])

    #rhouv = np.array([lf.rhol(q=q) for q in qq])
    rho_of_z = np.zeros([len(qq), len(veff.zgrid)])
    for i, q in enumerate(qq):
        lf.set_parameters(q)
        dN_dV_dlogL = lf.evaluate(veff.loglgrid, veff.zgrid, in_dlogl=True)
        dL_dV = dN_dV_dlogL * veff.dlogl[:, None] * (veff.lgrid[:, None] * Maggie_to_cgs)
        rho_of_z[i] = dL_dV.sum(axis=0)

    from util import quantile
    rho_ptile = quantile(rho_of_z.T, [0.16, 0.5, 0.84], weights=np.exp(log_w))
    rfig, rax = pl.subplots()
    rax.plot(veff.zgrid, rho_of_z[np.argmax(log_like)], label="MAP", color="royalblue")
    rax.plot(veff.zgrid, rho_ptile[:, 1], label="median", linestyle=":", color="royalblue")
    rax.fill_between(veff.zgrid, rho_ptile[:, 0], y2=rho_ptile[:, -1], color="royalblue",
                     alpha=0.5, label="16th-84th percentile")
    rax.set_yscale("log")
    pl.savefig('test.png')