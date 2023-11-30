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
    parser.add_argument("--f_cover", type=float,
                        default=1)
    parser.add_argument("--zref", type=float,
                        default=14)
    #create parser
    args = parser.parse_args()

    #print important arguments to the screen
    if(args.verbose):
        print(f'Number of samples {args.n_samples}')
        print(f'Minimum redshift {args.zmin}')
        print(f'Maximum redshift {args.zmax}')
        print(f'Minimum Muv {args.loglmin*-2.5}')
        print(f'Maximum Muv {args.loglmax*-2.5}')
        print(f'Fraction of uncovered pixels {args.f_cover}')

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
                                      comp_det=comp_det, comp_sel=comp_sel, f_cover=args.f_cover)

    #save a figure
    plot_veff(loglgrid,zgrid,veff)

    # ---------------------------
    # --- Set up model and priors
    # ---------------------------
    lf = EvolvingSchechter(zref=args.zref)
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
        if(args.evolving==2):
            param_names = ["phi0", "phi1", "lstar0", "alpha"]
        else:
            param_names = ["phi0", "phi1", "lstar0", "lstar1", "alpha"]
    else:
        param_names = ["phi0", "lstar0", "alpha"]
    params = Parameters(param_names, pdict)

    # -------------------
    # --- Initial ---
    # -------------------
    q_true = np.array([-4, 0, (21 / 2.5), 0, -1.7])
    if args.evolving:
        if(args.evolving==2):
            qq_true = np.array([q_true[0], q_true[1], q_true[2], q_true[4]])
        else:
            qq_true = q_true

    else:
        qq_true = np.array([q_true[0], q_true[2], q_true[4]])

    # -------------------
    # --- lnprobfn -------
    # -------------------
    assert np.isfinite(params.prior_product(qq_true))
    lnprobfn = partial(lnlike, data=jof, lf=lf, veff=veff, evolving=args.evolving)

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


    # -------------------
    # --- fitting -------
    # -------------------
    from infer import fit
    if args.fitter == "nautilus":
        lnprob = lnprobfn_dict
    else:
        lnprob = lnprobfn

    result = fit(params, lnprob, fitter=args.fitter, verbose=args.verbose)
    points, log_w, log_like, sampler = result

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
    if(args.evolving==2):
        sample_table['phistar'] = points[:,0]
        sample_table['dphidz']  = points[:,1]
        sample_table['mstar']   = -2.5*points[:,2]
        sample_table['alpha']   = points[:,3]
    else:
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