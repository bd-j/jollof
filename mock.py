"""mock.py - generate a mock effective volume and lf, sample from that, and then
infer the LF parameters from the samples.
"""

from functools import partial
import numpy as np
import matplotlib.pyplot as pl
from astropy.table import Table

from lf import create_parser
from lf import EvolvingSchechter
from lf import EffectiveVolumeGrid, construct_effective_volume
from lf import lum_to_mag, mag_to_lum, arcmin

from data import DataSamples
from util import sample_twod
from priors import Parameters, Uniform, Normal
from infer import lnlike, transform, fit

maggies_to_nJy = 3631e9


def make_mock(loglgrid, zgrid, omega, q_true,
              lf=EvolvingSchechter(),
              n_samples=100,
              sigma_logz=0.1,
              sigma_flux=1/maggies_to_nJy,
              completeness_kwargs={},
              selection_kwargs={},
              fake_flag=True,
              seed=None):
    """Draw a set of mock galaxies from a LF x Veff, add noise to them and draw
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


if __name__ == "__main__":

    parser = create_parser()
    parser.add_argument("--fitter", type=str, default="none",
                        choices=["none", "dynesty", "emcee", "brute",
                                 "ultranest", "nautilus"])
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--evolving", type=int, default=0)
    parser.add_argument("--regenerate_mock", type=int, default=1)
    parser.add_argument("--sample_output", type=str, default='./output/mock_posterior_samples.fits')

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
    lf = EvolvingSchechter(zref=args.zref)
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
    q_true = np.array([-4, -0.2, (21 / 2.5), 0, -2.0])
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
    if args.regenerate_mock:
        mock, veff = make_mock(loglgrid, zgrid, args.omega,
                            q_true, lf=lf,
                            sigma_logz=0.05,
                            sigma_flux=0.1/maggies_to_nJy,
                            n_samples=args.n_samples,
                            fake_flag=args.fake_flag)
        print(f"{len(mock.all_samples)} objects drawn from this LF x Veff")
        dN, _ = lf.n_effective(veff)

        fig, ax = pl.subplots()
        dens = np.log10(dN)
        cb = ax.imshow(dens, origin="lower", cmap="Blues", alpha=0.8,
                       vmax=dens.max(), vmin=dens.max()-5,
                    extent=[zgrid.min(), zgrid.max(), loglgrid.min(), loglgrid.max()],
                    aspect="auto")
        _, ax = mock.show(ax=ax)
        fig.colorbar(cb, label="log(dN)")
        fig.savefig("./output/mock_samples.png", dpi=300)
        mock.to_fits("./output/mock_data.fits")
        veff.to_fits("./output/mock_veff.fits")
    else:
        mock = DataSamples(filename="./output/mock_data.fits", n_samples=args.n_samples)
        veff = EffectiveVolumeGrid(fromfitsfile="./output/mock_veff.fits")

    #sys.exit()

    # ----------------
    # --- lnprobfn ---
    # ----------------
    assert np.isfinite(params.prior_product(qq_true))
    lnprobfn = partial(lnlike, data=mock, lf=lf, veff=veff, evolving=args.evolving)

    # for nautilus
    if (args.evolving == 2):
        def lnprobfn_dict(param_dict):
            qq = np.array([param_dict['phi0'], param_dict['phi1'],
                           param_dict['lstar0'],
                           param_dict['alpha']])
            return lnprobfn(qq)
    elif (args.evolving == 1):
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
    fig = corner.corner(points, weights=np.exp(log_w), bins=20,
                        labels=params.free_params,
                        plot_datapoints=False, plot_density=False,
                        fill_contours=True, levels=(0.68, 0.95),
                        range=np.ones(ndim) * 0.999, fig=fig,
                        truths=qq_true, truth_color="red")

    fig.suptitle(f"Mock: {qq_true}")
    fig.savefig(f"posteriors-{args.fitter}.png", dpi=300)
    print(f"MAP={points[np.argmax(log_like)]}")

    # - luminosity density -
    n = -5000  #restrict to samples with non-negligible weights, for speed
    qq = np.array([transform(p, evolving=args.evolving) for p in points[n:]])

    lmin, lmax, nlx = 6.8, 20, 100
    rho_uv_array = np.zeros([len(qq), len(zgrid)])
    for k, q in enumerate(qq):
        rho_uv_array[k, :] = lf.rhol(zgrid, q, lmin, lmax, nlx)

    from util import quantile
    rho_ptile = quantile(rho_uv_array.T, [0.16, 0.5, 0.84], weights=np.exp(log_w[n:]))
    rfig, rax = pl.subplots()
    rax.plot(zgrid, rho_uv_array[np.argmax(log_like[n:])], label="MAP", color="royalblue")
    rax.plot(zgrid, rho_ptile[:, 1], label="median", linestyle=":", color="royalblue")
    rax.fill_between(veff.zgrid, rho_ptile[:, 0], y2=rho_ptile[:, -1], color="royalblue",
                     alpha=0.5, label="16th-84th percentile")
    rax.set_yscale("log")

    # ---------------
    # Save samples to a fits file
    # ---------------
    sample_table = Table()
    sample_table['phistar'] = points[:,0]
    sample_table['mstar']   = -2.5*points[:,1]
    sample_table['alpha']   = points[:,2]
    sample_table['loglike'] = log_like
    sample_table['logw']    = log_w
    sample_table.write(args.sample_output, format='fits', overwrite=True)
