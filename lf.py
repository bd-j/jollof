import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
from astropy.io import fits

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
        default=1000,
        metavar='nz',
        type=int,
        help='Number of redshift samples (default: 1000)')

    #Minimum log L
    parser.add_argument('--loglmin',
        default=15/2.5,
        metavar='loglmin',
        type=float,
        help=f'Minimum log luminosity to model (default: {15/2.5})')

    #Maximum log L
    parser.add_argument('--loglmax',
        default=22/2.5,
        metavar='loglmax',
        type=float,
        help=f'Maximum log luminosity to model (default: {22/2.5})')

    #Luminosity samples
    parser.add_argument('--nl',
        default=1000,
        metavar='nl',
        type=int,
        help='Number of luminosity samples (default: 1000)')

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
def schechter(logl, logphi, loglstar, alpha, l_min=None):
    """
    Generate a Schechter function (in dlogl).
    """
    phi = ((10**logphi) * np.log(10) * 10**((logl - loglstar) * (alpha + 1)) * np.exp(-10**(logl - loglstar)))
    return phi


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

    def evaluate(self, L, z):
        self.set_redshift(z)
        x = (L / self.lstar)
        return self.phi * x**self.alpha * np.exp(-x)


    # record the LF parameter evolution
    # to an ascii table
    def record_parameter_evolution(self, zgrid):

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

    # record the LF evolution to
    # a fits image
    def record_lf_evolution(self, lgrid, zgrid):
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

    # plot the LF evolution
    def plot_lf_evolution(self, lgrid, zgrid):
        # get the LF grid
        lf = self.evaluate(lgrid, zgrid)

        # make a figure
        cmap = colormaps.get_cmap('turbo')
        f, ax = plt.subplots(1, 1, figsize=(7, 7))
        for i in range(len(zgrid)):
            izg = (zgrid[i] - zgrid[0]) / (zgrid[-1] - zgrid[0])
            ax.plot(np.log10(lgrid), lf[:, i], color=cmap(izg))
        ax.set_xlim([np.log10(lgrid[0]), np.log10(lgrid[-1])])
        ax.set_ylim([1.0e-7, 1])
        ax.set_xlabel('log10 Luminosity')
        ax.set_ylabel(r'$\phi(L)$')
        ax.set_yscale('log')
        plt.savefig('lf_evolution.png', bbox_inches='tight', facecolor='white')


#########################################
# sigmoid to model completeness
#########################################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#########################################
# Effective volume vs. z and mag
#########################################
def effective_volume(mag, zgrid, lgrid, omega, mag_50=30, dc=1):
    """compute this on a grid of z and Muv
    """
    volume = omega * cosmo.differential_comoving_volume(zgrid).value
    muv = lgrid + cosmo.distmod(zgrid).value
    completeness_function = sigmoid((mag_50 - muv) / dc)  # fake completeness function
    selection_function = completeness_function * 1.0

    return selection_function * volume, zgrid, lgrid

#########################################
# Log likelihood L(data|q)
#########################################
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
    zgrid = np.linspace(args.zmin, args.zmax, args.nz)[None, :]

    # grid of log luminosity
    loglgrid = np.linspace(args.loglmin, args.loglmax, args.nl)[:, None]

    # luminosity
    lgrid = 10**loglgrid

    # initialize evolving schechter
    # q_true = np.array([0.0, 0.0, 1e-3, 0, 0, 10**(18 / 2.5), -1.5])
    q_true = np.array([0.0, -1.0e-4, 1e-3, 0, 0, 10**(18 / 2.5), -1.5])
    s = EvolvingSchechter()
    s.set_parameters(q_true)

    # write the parameter evolution to a file
    if (args.verbose):
        print('Recording parameter evolution with redshift...')
    s.record_parameter_evolution(zgrid[0])

    # write the evolving luminosity function
    # to a binary file
    if (args.verbose):
        print('Writing luminosity function evolution to file...')
    s.record_lf_evolution(lgrid, zgrid[0])

    # plot the luminosity function evolution
    # and save as a png
    if (args.verbose):
        print('Plotting luminosity function evolution...')
    s.plot_lf_evolution(lgrid, zgrid[0])

    # schechter function evaluated on a grid of z and L
    schechter = s.evaluate(lgrid, zgrid[0])

    # sampling
    ns = 1000
    sflat = schechter.flatten()
    sind = np.arange(len(sflat))
    inds = np.random.choice(sind, p=sflat/np.nansum(sflat), size=ns)
    xx = inds // args.nz
    yy = np.mod(inds, args.nz)
    zs = zgrid[0][yy]
    loglums = loglgrid[xx]

    #plt.ion()
    fig, ax = plt.subplots()
    ax.imshow(np.log10(schechter), origin="lower", cmap="Blues", alpha=0.4,
              extent=[zgrid.min(), zgrid.max(), loglgrid.min(), loglgrid.max()])
    ax.plot(zs, loglums, "o", color="red", label="samples")
    ax.set_ylim(args.loglmin, args.loglmax)
    ax.set_xlabel("redshift")
    ax.set_ylabel("log L")
    fig.savefig("lf_samples.png")
    #done!
    print('Done!')

########################
# Run the program
########################
#if __name__ == "__main__":
#    main()

