import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
from astropy.io import fits
from astropy.units import arcmin

#########################################
# Routine to parse command line arguments
#########################################
def create_parser():

    # Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="Model an evolving schchter function.")

    #Minimum redshift
    parser.add_argument('--zmin',
        default=12.0,
        metavar='zmin',
        type=float,
        help='Minimum redshift to model (default: 12)')

    #Maximum redshift
    parser.add_argument('--zmax',
        default=16.0,
        metavar='zmax',
        type=float,
        help='Maximum redshift to model (default: 16)')

    #Redshift samples
    parser.add_argument('--nz',
        default=400,
        metavar='nz',
        type=int,
        help='Number of redshift samples (default: 1000)')

    #Minimum log L
    parser.add_argument('--loglmin',
        default=16/2.5,
        metavar='loglmin',
        type=float,
        help=f'Minimum log luminosity (absolute maggies) to model (default: {15/2.5})')

    #Maximum log L
    parser.add_argument('--loglmax',
        default=24/2.5,
        metavar='loglmax',
        type=float,
        help=f'Maximum log luminosity (absolute maggies) to model (default: {22/2.5})')

    #Luminosity samples
    parser.add_argument('--nl',
        default=1000,
        metavar='nl',
        type=int,
        help='Number of luminosity samples (default: 1000)')

    parser.add_argument('--lf_params',
        nargs='+',
        default=[0.0, 0.0, 1e-3, 0, 0, 10**(21 / 2.5), -1.5],
        type=float,
        help='list of aperture diameters (in arcsec)')

    #Area in arcmin^2
    parser.add_argument('--area',
        default= 9.05,
        metavar='area',
        type=float,
        help=f'Area in arcmin^2 (default: 9.05)')

    #Verbosity
    parser.add_argument('-v', '--verbose',
        dest='verbose',
        action='store_true',
        help='Print helpful information to the screen? (default: False)',
        default=False)

    return parser

#########################################
# Schechter function in log_10 L
#########################################
def log_schechter(logl, logphi, loglstar, alpha, l_min=None):
    """
    Generate a Schechter function (in dlogl).
    """
    phi = ((10**logphi) * np.log(10) * 10**((logl - loglstar) * (alpha + 1)) * np.exp(-10**(logl - loglstar)))
    return phi

#########################################
# Convert luminosity to magnitude
#########################################
def lum_to_mag(logl, zred):
    mag = -2.5 * logl + cosmo.distmod(zred).value -2.5*np.log10(1+zred)
    return mag


class EvolvingSchechter:
    """ Class to model an evolving schechter fnc
    """
    def __init__(self, zref=14, order=2):
        self.zref = zref
        self.order = 2

        # determines mapping from theta vector to parameters
        self._phi_index = [0, 1, 2]
        self._lstar_index = [3, 4, 5]
        self._alpha_index = [6]

    def set_parameters(self, q):
        self._phis = q[self._phi_index]
        self._lstars = q[self._lstar_index]
        self._alphas = q[self._alpha_index]

    def set_redshift(self, z):
        zz = z - self.zref
        # print(f'zz.shape {zz.shape}')

        # by default, vander decreases order
        # with increasing index
        self.phi = np.dot(np.vander(zz, len(self._phis)), self._phis, )
        self.lstar = np.dot(np.vander(zz, len(self._lstars)), self._lstars)
        self.alpha = np.dot(np.vander(zz, len(self._alphas)), self._alphas)
        # print(f'self.phi.shape {self.phi.shape}')

    def evaluate(self, L, z, grid=True, in_dlogl=False):
        self.set_redshift(z)
        if grid:
            x = (L[:, None] / self.lstar)
        else:
            x = (L / self.lstar)
        if in_dlogl:
            factor = np.log(10)
        else:
            factor = 1
        return factor * self.phi * x**(self.alpha + int(in_dlogl)) * np.exp(-x)

    def record_parameter_evolution(self, zgrid):
        """record the LF parameter evolution to an ascii table
        """
        # set the redshift evolution
        # of LF parameters
        self.set_redshift(zgrid)

        # make a table and save to file
        t = Table()
        t['z'] = zgrid
        t['phi'] = self.phi
        t['lstar'] = self.lstar
        t['alpha'] = self.alpha
        t.write('lf_parameter_evolution.txt', format='ascii', overwrite=True)

    def record_lf_evolution(self, lgrid, zgrid):
        """record the LF evolution to a fits image
        """
        # get the LF grid
        lf = self.evaluate(lgrid, zgrid)

        # make a column and bin table for luminosity
        coll = fits.Column(name='luminosity', format='E', array=lgrid)
        lhdu = fits.BinTableHDU.from_columns([coll], name='luminosity')

        # make a column and bin table for redshift
        colz = fits.Column(name='redshift', format='E', array=zgrid)
        zhdu = fits.BinTableHDU.from_columns([colz], name='redshift')

        # record the lf as an image (2D)
        lfhdu = fits.ImageHDU(name='lf', data=lf)

        # write to a fits file
        phdu = fits.PrimaryHDU()
        hdul = fits.HDUList([phdu, zhdu, lhdu, lfhdu])
        hdul.writeto('lf_evolution.fits', overwrite=True)

    def plot_lf_evolution(self, lgrid, zgrid, in_dlogl=False):
        """plot the LF evolution
        """
        # get the LF grid
        lf = self.evaluate(lgrid, zgrid, in_dlogl=in_dlogl)
        x = -2.5 * np.log10(lgrid)

        # make a figure
        cmap = colormaps.get_cmap('turbo')
        f, ax = plt.subplots(1, 1, figsize=(7, 7))
        for i in range(len(zgrid)):
            izg = (zgrid[i] - zgrid[0]) / (zgrid[-1] - zgrid[0])
            ax.plot(x, lf[:, i], color=cmap(izg))
#        ax.set_xlim([x[0], x[-1]])
#        ax.set_ylim([1.0e-7, 1])
        ax.set_xlim([-23,-18])  #ouchi 2009 fig 7
        ax.set_ylim([4e-7,1e-2])  #ouchi 2009 fig 7
        ax.set_xlabel(r'M$_{\rm UV}$')
        ax.set_ylabel(r'$\phi(L)$')
        ax.set_yscale('log')
        ax.tick_params(which='both', direction='in', right=True)
        plt.savefig('lf_evolution.png', bbox_inches='tight', facecolor='white')


# ------------------------------
# sigmoid to model completeness
# ------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ------------------------------
# Effective volume vs. z and mag
# ------------------------------
def effective_volume(loglgrid, zgrid, omega,
                     completeness_kwargs={},
                     selection_kwargs={}):
    """compute this on a grid of z and Muv

    Returns
    -------
    dV_dz : array of shape N_lum, Nz
       Effective volume in each luminosity and redshift bin.  This is the
       differential volume in Mpc^3/redshift, multiplied by the probability of
       an object at that luminsity and redshift being in the catalog.
    """
    # dV/dz (Mpc^3/redshift)
    volume = omega * cosmo.differential_comoving_volume(zgrid).value
    muv = lum_to_mag(loglgrid[:, None], zgrid)
    # fake completeness function
    completeness = completeness_function(muv, **completeness_kwargs)
    # fake selection function
    selection_function = completeness * (muv < 31)

    return selection_function * volume

# --------------------
# --- Completeness ---
# --------------------
def completeness_function(mag, mag_50=30, dc=0.5):
    completeness = sigmoid((mag_50 - mag) / dc)
    return completeness

# ------------------------
# Log likelihood L(data|q)
# ------------------------
def lnlike(q, data, effective_volume):
    veff, zgrid, Muv_grid = effective_volume
    lgrid = 10**(-0.4 * Muv_grid)  # units are now absolute maggies
    s = EvolvingSchechter()
    s.set_parameters(q)

    # if data likelihoods are evaluated on the same grid
    schechter = s.evaluate(lgrid, zgrid)
    N_theta = integrate(np.ones_like(schechter), schechter, veff)
    lnlike = np.zeros(len(data))
    for i, d in enumerate(data):
        like = integrate(d.probability, schechter, veff)
        lnlike[i] = np.log(like)

    return np.sum(lnlike) - np.log(N_theta)


# ------------------------------
# Sample from a 2d histogram
# ------------------------------
def sample_twod(X, Y, Z, n_sample=1000):

    sflat = Z.flatten()
    sind = np.arange(len(sflat))
    inds = np.random.choice(sind, size=n_sample,
                            p=sflat / np.nansum(sflat))
    # TODO: check x, y order here
    N = len(np.squeeze(Y))
    xx = inds // N
    yy = np.mod(inds, N)

    y = np.squeeze(Y)[yy]
    x = np.squeeze(X)[xx]
    return x, y


########################
# The main function
########################
#def main():
if __name__ == "__main__":

    # create the command line argument parser
    parser = create_parser()

    # store the command line arguments
    args = parser.parse_args()

    # grid of redshifts
    zgrid = np.linspace(args.zmin, args.zmax, args.nz)

    # grid of log luminosity
    loglgrid = np.linspace(args.loglmin, args.loglmax, args.nl)

    # luminosity
    lgrid = 10**loglgrid

    # absolute magnitude
    Muvgrid = -2.5 * loglgrid

    # initialize evolving schechter
    # q_true = np.array([0.0, 0.0, 1e-3, 0, 0, 10**(18 / 2.5), -1.5])
    print(args.lf_params)
    #exit()
    #q_true = np.array([0.0, -1.0e-4, 1e-3, 0, 0, 10**(21 / 2.5), -1.5])
    q_true = np.array(args.lf_params)
    s = EvolvingSchechter()
    s.set_parameters(q_true)

    # write the parameter evolution to a file
    if (args.verbose):
        print('Recording parameter evolution with redshift...')
    s.record_parameter_evolution(zgrid)

    # write the evolving luminosity function
    # to a binary file
    if (args.verbose):
        print('Writing luminosity function evolution to file...')
    s.record_lf_evolution(lgrid, zgrid)

    # plot the luminosity function evolution
    # and save as a png
    if (args.verbose):
        print('Plotting luminosity function evolution...')
    s.plot_lf_evolution(lgrid, zgrid, in_dlogl=True)

    # sample LF
    if (args.verbose):
        print('sampling from the LF...')
    lf = s.evaluate(lgrid, zgrid)
    loglums, zs = sample_twod(loglgrid, zgrid, lf, n_sample=1000)
    fig, ax = plt.subplots()
    ax.imshow(np.log10(lf), origin="lower", cmap="Blues", alpha=0.5,
              extent=[zgrid.min(), zgrid.max(), Muvgrid.max(), Muvgrid.min()],
              aspect="auto")
    ax.plot(zs, -2.5 * loglums, "o", color="red", label="samples")
    ax.set_ylim(-2.5*args.loglmin, -2.5*args.loglmax)
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"M$_{\rm UV}$")
    fig.savefig("lf_samples.png")

    # sample number counts
    if (args.verbose):
        print('sampling from the number counts...')
    omega = (args.area * arcmin**2).to("steradian").value
    lf = s.evaluate(lgrid, zgrid, in_dlogl=True)
    veff = effective_volume(loglgrid, zgrid, omega)
    dN_dz_dlogl = lf * veff
    dN = dN_dz_dlogl * np.gradient(zgrid) * np.gradient(loglgrid)[:, None]
    N_bar = dN.sum()
    N = np.random.poisson(N_bar, size=1)[0]
    print(f"Drew {N} galaxies from expected total of {N_bar}")
    loglums, zs = sample_twod(loglgrid, zgrid, dN, n_sample=N)
    fig, ax = plt.subplots()
    ax.imshow(dN, origin="lower", cmap="Blues", alpha=0.5,
              extent=[zgrid.min(), zgrid.max(), Muvgrid.max(), Muvgrid.min()],
              aspect="auto")
    ax.plot(zs, -2.5 * loglums, "o", color="red", label="samples")
    ax.set_ylim(-2.5*args.loglmin, -2.5*args.loglmax)
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"M$_{\rm UV}$")
    fig.savefig("lf_samples.png")

    #done!
    print('Done!')

########################
# Run the program
########################
#if __name__ == "__main__":
#    main()

