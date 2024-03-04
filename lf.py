import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.interpolate import RegularGridInterpolator
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
from astropy.io import fits
from astropy.units import arcmin

from util import sample_twod

# TODO:  rewrite everything in jax.

__all__ = ['EvolvingSchechter', 'EvolvingSchechterExp',
           'EffectiveVolumeGrid', 'CompletenessGrid',
           'lum_to_mag', 'mag_to_lum',
           ]


#########################################
# Routine to parse command line arguments
#########################################
def create_parser():

    # Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="Model an evolving schchter function.")

    #Minimum redshift
    parser.add_argument('--zmin',
        default=11.5,
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
        help=f'Minimum log luminosity (absolute maggies) to model (default: {16/2.5})')

    #Maximum log L
    parser.add_argument('--loglmax',
        default=24/2.5,
        metavar='loglmax',
        type=float,
        help=f'Maximum log luminosity (absolute maggies) to model (default: {24/2.5})')

    #Luminosity samples
    parser.add_argument('--nl',
        default=1000,
        metavar='nl',
        type=int,
        help='Number of luminosity samples (default: 1000)')

    #Evovling luminosity function parameters
    parser.add_argument('--lf_params',
        nargs=5,
        default=[-4, 0, (21 / 2.5), 0, -1.8],
        type=float,
        help='LF parameters')

    #Area in arcmin^2
    parser.add_argument('--area',
        default= 9.05,
        metavar='area',
        type=float,
        help=f'Area in arcmin^2 (default: 9.05)')

    #Complete
    parser.add_argument('--complete',
        dest='complete',
        action='store_true',
        help='Model LF as complete? (default: False)',
        default=False)

    parser.add_argument('--fake_flag',
        dest='fake_flag',
        action='store_true',
        help='Model LF with a fake completeness function?',
        default=False)

    #Verbosity
    parser.add_argument('-v', '--verbose',
        dest='verbose',
        action='store_true',
        help='Print helpful information to the screen? (default: False)',
        default=False)

    parser.add_argument("--zref", type=float,
                        default=14)

    return parser


#########################################
# Convert luminosity to magnitude and vice versa
#########################################
Maggie_to_cgs = 4.344211434763621e20


def lum_to_mag(logl, zred):
    mag = -2.5 * logl + cosmo.distmod(zred).value - 2.5*np.log10(1+zred)
    return mag


def mag_to_lum(mag, zred):
    logl = -0.4 * (mag - cosmo.distmod(zred).value + 2.5*np.log10(1+zred))
    return logl


class EvolvingDF:
    """ Class to model an evolving schechter fnc

    phi : N/Mpc^3/luminosity at lstar
    lstar: luminosity of the exponential cutoff, in absolute maggies (i.e., 10**(-0.4M_{AB}))
    alpha : power law slope below lstar

    The default parameter mapping is:
        [phi0, phi1, lstar0, lstar1, alpha]


    This can be altered by changing the `_<param>_index` arrays.  The evolution
    is generally computed with respect to a reference redshift, `zref`
    """
    def __init__(self, zref=14):
        self.zref = zref
        self.params = [("phi", [0, 1]),
                       ("lstar", [2, 3]),
                       ("alpha", [4])
                       ]

    def set_parameters(self, q):
        """This picks out the parameters and their evolution from a one
        dimensional vector, and stores them in internal arrays

        Parameters
        ----------
        q : ndarray of shape (npar,)
        """
        for p, inds in self.params:
            setattr(f"_{p}", q[inds])

    def set_redshift(self, z=None):
        """Use the model parameter values to compute the LF parameters at the
        given redshifts, and cache them.

        Parameters
        ----------
        z : optional, scalar or ndarray
            The redshifts at which LF parameters should be computed

        """
        if z is None:
            z = self.zref
        zz = z-self.zref
        for p, inds in self.params:
            _par = getattr(self, f"_{p}")
            value, order = 0, len(inds)
            for i in range(order):
                value += _par[i] * zz**i
            setattr(self, p, value)

    def evaluate(self, L, z, grid=True, in_dlogl=False):
        """
        Returns
        -------
        dN/(dVdL) where dV is Mpc^3 and dL is in absolute Maggies

        or

        dN/(dVdlogL) where dV is Mpc^3 and dlogL is base log_10(L)

        """
        self.set_redshift(z)
        raise(NotImplementedError)

    def n_effective(self, veff):
        """Compute the expected number of objects given a selection function on
        a grid of logL and redshift.

        Parameters
        ----------
        veff : instance of EffectiveVolmeGrid

        Returns
        -------
        dN : ndarray of shape (N_L, N_z)
            The number of galaxies expected to be observed in each bin of logL and redshift

        dV : ndarray of shape (N_L, N_z)
            The effective volume (in Mpc^3 * 1) in each bin of logL and redshift.
        """
        # this takes most of the time:
        lf = self.evaluate(veff.lgrid, veff.zgrid, grid=True, in_dlogl=True)
        dV_dz = veff.data  # shape (nL, nz)

        # TODO: use dot products here for the integral
        dV = dV_dz * veff.dz[None, :]
        dN = lf * dV * veff.dlogl[:, None]

        return dN, dV

    def rhol(self, zgrid, q=None, lmin=6.8, lmax=20.0, nlx=100):
        """Compute the integrated luminosity density
        between lmin and lmax as a function of redshift

        Parameters
        -----------
        zgrid : ndarray of shape (n_z,) or scalar
            Redshifts at which to compurte the luminosity density

        q : optional, ndarray of shape ndim,
            The full set of LF parameters.

        Returns
        -------
        rho: ndarry of shape (nz,) or scalar
            L/Mpc^3 in erg/s/Hz/Mpc^3
        """
        #set parameters if needed
        if q is not None:
            self.set_parameters(q)
        zz = np.atleast_1d(zgrid)
        logl = np.linspace(lmin, lmax, nlx)
        Luv = 10**logl * Maggie_to_cgs
        rho_integrated = np.zeros(len(zz))
        phi = self.evaluate(10**logl, zz, in_dlogl=True)
        for i, z in enumerate(zz):
            rho_integrated[i] = np.trapz(phi[:, i] * Luv, x=np.log10(Luv))
        return np.squeeze(rho_integrated)

    def nl(self, z=None, q=None, lmin=7.0, lmax=20.0, nlx=100):
        """Compute the integrated number density
        between lmin and lmax as a function
        of redshift

        Returns
        -------
        n/Mpc^3
        """

        #set parameters if needed
        if q is not None:
            self.set_parameters(q)

        #array of luminosity and phi
        l_array   = np.linspace(lmin,lmax,nlx)
        phi_array = np.array(self.evaluate(10**l_array, z, grid=False, in_dlogl=True))

        #return n
        n_integrated = np.trapz(phi_array, x=l_array)

        return n_integrated #1/Mpc^3

    def nM(self, z=None, q=None, Mmin=-22.3, Mmax=-18.0, nMx=100):
        """Compute the integrated number density between Mmin and Mmax absolute
        magnitudes as a function of redshift

        Returns
        -------
        n/Mpc^3
        """

        #set parameters
        if q is not None:
            self.set_parameters(q)

        #set redshift in evolving lF
        self.set_redshift(z)

        #set min and max luminosities
        #based on min and max magnitues
        lmin = Mmax/-2.5
        lmax = Mmin/-2.5

        #return nl
        return self.nl(z=z,q=q,lmin=lmin,lmax=lmax,nlx=nMx)

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

    def plot_lf_evolution(self, lgrid, zgrid, in_dlogl=False,
                          filename='./output/lf_evolution.png'):
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
        ax.set_xlim([-23, -18])  #ouchi 2009 fig 7
        ax.set_ylim([4e-7, 1e-2])  #ouchi 2009 fig 7
        ax.set_xlabel(r'M$_{\rm UV}$')
        ax.set_ylabel(r'$\phi(L)$')
        ax.set_yscale('log')
        ax.tick_params(which='both', direction='in', right=True)
        plt.savefig(filename, bbox_inches='tight', facecolor='white')

    def sample_lf(self, loglgrid, zgrid, n_sample=100,
                  filename="./output/lf_samples.png"):
        """Draw some samples from the LF(L, z) distribution and plot them.
        """
        lf = s.evaluate(10**loglgrid, zgrid)
        loglums, zs = sample_twod(loglgrid, zgrid, lf, n_sample=n_sample)
        fig, ax = plt.subplots()
        ax.imshow(np.log10(lf), origin="lower", cmap="Blues", alpha=0.5,
                  extent=[zgrid.min(), zgrid.max(), Muvgrid.max(), Muvgrid.min()],
                  aspect="auto")
        ax.plot(zs, -2.5 * loglums, "o", color="red", label="samples")
        ax.set_ylim(-2.5*loglgrid.min(), -2.5*loglgrid.max())
        ax.set_xlabel("redshift")
        ax.set_ylabel(r"M$_{\rm UV}$")
        fig.savefig(filename)
        return loglums, zs


class Schechter:

    def __init__(self, zref=14, order=1):
        self.zref = zref
        self.order = order

        # determines mapping from theta vector to parameters
        self._phi_index = [0, 1]
        self._lstar_index = [2, 3]
        self._alpha_index = [4]

    def set_parameters(self, q):
        """This picks out the parameters and their evolution from a one
        dimensional vector, and stores them in internal arrays

        Parameters
        ----------
        q : ndarray of shape (npar,)
        """
        self._phis = q[self._phi_index]
        self._lstars = q[self._lstar_index]
        self._alphas = q[self._alpha_index]

    def set_redshift(self, z=None):
        """Use the model parameter values to compute the LF parameters at the
        given redshifts, and cache them.

        Parameters
        ----------
        z : optional, scalar or ndarray
            The redshifts at which LF parameters should be computed

        """
        if z is None:
            z = self.zref

        # TODO: correct this
        # --- phi = phi_0 \, (1+z)^\beta ----
#        zz = np.log10((1 + z) / (1 + self.zref))
        zz = z-self.zref

        self.alpha = self._alphas[0]

        logphi = self._phis[0] + self._phis[1] * zz
        self.phi = 10**logphi

        loglstar = self._lstars[0] + self._lstars[1] * zz
        self.lstar = 10**loglstar

    def evaluate(self, L, z, grid=True, in_dlogl=False):
        """
        Returns
        -------
        dN/(dVdL) where dV is Mpc^3 and dL is in absolute Maggies

        or

        dN/(dVdlogL) where dV is Mpc^3 and dlogL is base log_10(L)

        """
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


class EvolvingSchechterExp(EvolvingDF, Schechter):

    def set_redshift(self, z=None):
        """Use the model parameter values to compute the LF parameters at the
        given redshifts, and cache them
        """
        if z is None:
            z = self.zref

        self.alpha = self._alphas

        # TODO: correct this
        # --- phi = phi_0 \, \exp{\beta \, \Delta z} ---
        zz = z - self.zref

        logphi = self._phis[0] + self._phis[1] * zz
        self.phi = 10**logphi

        loglstar = self._lstars[0] + self._lstars[1] * zz
        self.lstar = 10**loglstar


class EvolvingSchecterPoly(EvolvingDF, Schechter):

    def __init__(self, zref=14, order=2):
        self.zref = zref
        self.order = order

        # determines mapping from theta vector to parameters
        self._phi_index = slice(0, order)
        self._lstar_index = slice(order, 2*order)
        self._alpha_index = slice(2*order, None)

    def set_redshift(self, z=None):
        """Use the model parameter values to compute the LF parameters at the
        given redshifts, and cache them
        """
        if z is None:
            z = self.zref

        # TODO: correct this
        # --- phi = pho0 + a_1 \, \Delta z + a_2 \, (\Delta z)^2---
        zz = z - self.zref
        # print(f'zz.shape {zz.shape}')
        # by default, vander decreases order
        # with increasing index
        self.phi = np.dot(np.vander(zz, len(self._phi_index)), self._phis)
        self.lstar = np.dot(np.vander(zz, len(self._lstar_index)), self._lstars)
        self.alpha = np.dot(np.vander(zz, len(self._alpha_index)), self._alphas)
        # print(f'self.phi.shape {self.phi.shape}')


class EvolvingDoublePowerLaw(EvolvingDF):

    def __init__(self, zref=14, order=1):
        self.zref = zref
        self.order = order

        # determines mapping from theta vector to parameters
        self._phi_index = [0, 1]
        self._lstar_index = [2, 3]
        self._alpha_index = [4] # evolving alpha?
        self._beta_index = [5]

    def set_parameters(self, q):
        """This picks out the parameters and their evolution from a one
        dimensional vector, and stores them in internal arrays

        Parameters
        ----------
        q : ndarray of shape (npar,)
        """
        self._phis = q[self._phi_index]
        self._lstars = q[self._lstar_index]
        self._alphas = q[self._alpha_index]  # faint-end slope
        self._betas = q[self._beta_index]  # bright end slope
        self._gamma = 1 # speed of transition

    def set_redshift(self):
        """Use the model parameter values to compute the LF parameters at the
        given redshifts, and cache them.

        Parameters
        ----------
        z : optional, scalar or ndarray
            The redshifts at which LF parameters should be computed

        """
        if z is None:
            z = self.zref
        # TODO: correct this
        # --- phi = phi_0 \, (1+z)^\beta ----
#        zz = np.log10((1 + z) / (1 + self.zref))
        zz = z-self.zref

        self.alpha = self._alphas[0]
        self.beta = self._betas[0]
        self.gamma = 1

        logphi = self._phis[0] + self._phis[1] * zz
        self.phi = 10**logphi

        loglstar = self._lstars[0] + self._lstars[1] * zz
        self.lstar = 10**loglstar

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
        faint = x**(self.alpha + int(in_dlogl))
        exp = (self.beta - self.alpha + int(in_dlogl)) / self.gamma
        bright = (1 + x**self.gamma)**exp

        return factor * self.phi *faint * bright

    def record_parameter_evolution(self, zgrid):
        pass

    def record_lf_evolution(self, lgrid, zgrid):
        pass

    def plot_lf_evolution(self):
        pass


class EffectiveVolumeGrid:
    """Thin wrapper on RegularGridInterpolator that keeps track of the grid points and grid values
    """
    def __init__(self, loglgrid=None, zgrid=None, veff=None,
                 fromfitsfile=None):

        if fromfitsfile is not None:
            loglgrid, zgrid, veff = self.from_fits(fromfitsfile)

        self.values = veff
        self.loglgrid = loglgrid
        self.lgrid = 10**loglgrid
        self.zgrid = zgrid
        self.dz = np.gradient(zgrid)
        self.dlogl = np.gradient(loglgrid)
        self.interp = RegularGridInterpolator((loglgrid, zgrid), veff,
                                              bounds_error=False,
                                              fill_value=0.0)

    def __call__(self, *args, **kwargs):
        return self.interp(*args, **kwargs)

    @property
    def data(self):
        return self.values

    def to_fits(self, fitsfilename):
        v = fits.ImageHDU(self.values, name="VEFF")
        v.header["UNITS"] = "differential effective volume in Mpc**3/redshift"
        l = fits.ImageHDU(-2.5*self.loglgrid, name="MUV")
        l.header["UNITS"] = "log_10(absolute maggies)"
        z = fits.ImageHDU(self.zgrid, name="Z")
        hdul = fits.HDUList([fits.PrimaryHDU(),
                             v, l, z])
        hdul.writeto(fitsfilename, overwrite=True)

    def from_fits(self, filename):
        with fits.open(filename) as hdul:
            veff = hdul["VEFF"].data[:]
            try:
                loglgrid = -0.4 * hdul["MUV"].data[:]
            except(KeyError):
                loglgrid = hdul["LOGL"].data[:]
            zgrid = hdul["Z"].data[:]
        return loglgrid, zgrid, veff


class CompletenessGrid:
    """Thin wrapper on RegularGridInterpolator that keeps track of the grid points and grid values
    """
    def __init__(self, comp, xgrid, ygrid, zgrid=None):
        self.values = comp
        self.xgrid  = xgrid
        self.ygrid  = ygrid
        self.dy = np.gradient(ygrid)
        self.dx = np.gradient(xgrid)
        if(zgrid is not None):
            self.zgrid = zgrid
            self.dz = np.gradient(zgrid)
            self.interp = RegularGridInterpolator((xgrid, ygrid, zgrid), comp,
                                              bounds_error=False,
                                              fill_value=0.0)
        else:
            self.interp = RegularGridInterpolator((xgrid, ygrid), comp,
                                              bounds_error=False,
                                              fill_value=0.0)

    def __call__(self, *args, **kwargs):
        return self.interp(*args, **kwargs)

    @property
    def data(self):
        return self.values


# ------------------------------
# sigmoid to model completeness
# ------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ------------------------------
# Effective volume vs. z and mag
# ------------------------------
def construct_effective_volume(loglgrid, zgrid, omega,
                     completeness_kwargs={},
                     selection_kwargs={},
                     fake_flag=False,
                     comp_sel=None,
                     comp_det=None,
                     muv_min=27, muv_max=32,
                     f_cover=1,
                     as_interpolator=True):
    """compute this on a grid of z and Muv

    Returns
    -------
    dV_dz : array of shape N_lum, Nz
       Effective volume in each luminosity and redshift bin.  This is the
       differential volume in Mpc^3/redshift, multiplied by the probability of
       an object at that luminsity and redshift being in the catalog.

        note for 3215, f_cover = 0.7785,

    """
    # Compute dV/dz (Mpc^3/redshift)
    volume = omega * cosmo.differential_comoving_volume(zgrid).value
    muv = lum_to_mag(loglgrid[:, None], zgrid) #apparent

    #use a fake completeness or simulated completeness?
    if(fake_flag):

        # Fake completeness function
        completeness = completeness_function(muv, **completeness_kwargs)
        #print(f'Completeness {completeness.shape} {completeness[0]}')

        # Fake selection function
        fsel = (muv < 31)
        zsel = sigmoid((zgrid - 11) / 0.25) * sigmoid((18 - zgrid) / 0.25)
        zsel = zsel * (zgrid > 9) * (zgrid < 20)
        selection_function = completeness * fsel * zsel[None, :]

        veff = selection_function * volume
        print(veff.shape)

    else:
        #initialize veff
        veff = volume

        #apply detection completeness
        if(comp_det is not None):

            #use an actual completeness function
            mab_clip_grid = np.clip(muv, muv_min, muv_max)

            #fake size
            #lrh = np.full_like(zgrid,-1.5)

            #apply completeness
            cd = comp_det((mab_clip_grid, -1.5)) #pretend all sizes are logrh<=-1.5
            veff = cd*veff # apply completeness
        else:
            veff = np.ones(len(loglgrid))[:, None] * veff

        #apply selection completeness
        if(comp_sel is not None):

            x,y = np.meshgrid(zgrid,-2.5*loglgrid[:, None])
            print(x.shape)
            print(y.shape)
            cs = comp_sel((x,y))

            #print(f'cs.shape {cs.shape}')
            #plt.clf()
            #plt.imshow(cs,origin='lower')
            #plt.savefig('test.png')
            #exit()

            #apply completeness
            veff = cs*veff # apply completeness
        else:
            veff = np.ones(len(loglgrid))[:, None] * veff

    #apply covering factor
    veff *= f_cover
    assert veff.min() >= 0

    #create an interpolator
    if as_interpolator:
        veff = EffectiveVolumeGrid(loglgrid, zgrid, veff)

    #return the grid or interpolator
    return veff


# --------------------
# --- Completeness ---
# --------------------
def completeness_function(mag, mag_50=30, dc=0.5, flag_complete=False):

    # Pretend we're completely complete
    if (flag_complete):
        return np.ones_like(mag)

    # Return a completeness vs. magnitude
    completeness = sigmoid((mag_50 - mag) / dc)
    return completeness


########################
# The main function
########################

if __name__ == "__main__":

    # Create the command line argument parser
    parser = create_parser()

    # Store the command line arguments
    args = parser.parse_args()

    # Output command line arguments
    if (args.verbose):
        print(f'Minimum redshift: {args.zmin}')
        print(f'Maximum redshift: {args.zmax}')
        print(f'Number of z bins: {args.nz}')
        print(f'Minimum log l   : {args.loglmin}')
        print(f'Maximum log l   : {args.loglmax}')
        print(f'Number of l bins: {args.nl}')
        print(f'Area in arcmin^2: {args.area}')
        print(f'Fully complete? : {args.complete}')

    # grid of redshifts
    zgrid = np.linspace(args.zmin, args.zmax, args.nz)

    # grid of log luminosity
    loglgrid = np.linspace(args.loglmin, args.loglmax, args.nl)

    # luminosity
    lgrid = 10**loglgrid

    # absolute magnitude
    Muvgrid = -2.5 * loglgrid

    # initialize evolving schechter
    print(args.lf_params)
    q_true = np.array(args.lf_params)
    s = EvolvingSchechter()
    s.set_parameters(q_true)

    # ----------------
    # write the parameter evolution to a file
    if (args.verbose):
        print('Recording parameter evolution with redshift...')
    s.record_parameter_evolution(zgrid)

    # --------------------------
    # write the evolving luminosity function
    # to a binary file
    if (args.verbose):
        print('Writing luminosity function evolution to file...')
    s.record_lf_evolution(lgrid, zgrid)

    # -----------------
    # plot the luminosity function evolution
    # and save as a png
    if (args.verbose):
        print('Plotting luminosity function evolution...')
    s.plot_lf_evolution(lgrid, zgrid, in_dlogl=True)

    # ------------
    # sample LF
    if (args.verbose):
        print('sampling from the LF...')
    loglums, zs = s.sample_lf(loglgrid, zgrid, n_sample=100)

    # --------------
    # sample number counts with a fake selection & completeness
    if (args.verbose):
        print('sampling from the number counts...')
    omega = (args.area * arcmin**2).to("steradian").value
    lf = s.evaluate(lgrid, zgrid, in_dlogl=True)
    completeness_kwargs = {'flag_complete': args.complete}
    veff = construct_effective_volume(loglgrid, zgrid, omega,
                                      completeness_kwargs,
                                      fake_flag=True,
                                      as_interpolator=True)

    dN, dV = s.n_effective(veff)
    N_bar = dN.sum()  # Average number of galaxies in survey
    V_bar = dV[0, :].sum()  # Effective volume of the survey in Mpc^3

    if (args.verbose):
        print(f'Area in arcmin^2 = {args.area}')
        print(f'Area in steraidians = {omega}')
        print(f'Cosmology: h = {cosmo.h}, Omega_m = {cosmo.Om0}')
        print(f'Effective volume = {V_bar} [Mpc^3].')
        print(f'Number density = {N_bar/V_bar} [Mpc^-3]')
        print(f'Number density (analytical) = {s.nl(zgrid, lmin=7.2, lmax=25.)} [Mpc^-3]')
        print(f'Luminosity density (analytical, MUV<-18) = {s.rhol(zgrid,lmin=7.2,lmax=25.) / 1e25} [10^25 erg s^-1 Hz^-1 Mpc^-3]')
        print(f'Luminosity density (analytical, Total) = {s.rhol(zgrid,lmin=-10,lmax=25.) / 1e25} [10^25 erg s^-1 Hz^-1 Mpc^-3]')

    # -----------------
    # Make a plot of the effective volume and the drawn galaxies
    N = np.random.poisson(N_bar, size=1)[0]
    note = f"Drew {N} galaxies from expected total of {N_bar:.2f}"
    if args.verbose:
        print(note)
    loglums, zs = sample_twod(loglgrid, zgrid, dN, n_sample=N)
    fig, axes = plt.subplots(2, 2, sharex='col', sharey='row',
                             gridspec_kw={"height_ratios":[1, 4], "width_ratios":[4, 1]})
    ax = axes[1, 0]
    ax.imshow(dN, origin="lower", cmap="Blues", alpha=0.5,
              extent=[zgrid.min(), zgrid.max(), Muvgrid.max(), Muvgrid.min()],
              aspect="auto")
    ax.plot(zs, -2.5 * loglums, "o", color="red", label="samples")
    ax.set_ylim(-2.5*args.loglmin, -2.5*args.loglmax)
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"M$_{\rm UV}$")
    ax = axes[0, 0]
    ax.hist(zs, bins=10, range=(zgrid.min(), zgrid.max()), density=True,
            alpha=0.5, color="tomato")
    ax.plot(veff.zgrid, dN.sum(axis=0)/dN.sum() / veff.dz, linestyle="--", color="royalblue")
    ax = axes[1, 1]
    ax.hist(-2.5*loglums, bins=10, range=(Muvgrid.min(), Muvgrid.max()), density=True,
            alpha=0.5, color="tomato", orientation="horizontal")
    #ax.plot(dN.sum(axis=-1)/dN.sum() /( dlogL * 2.5), Muvgrid, linestyle="--", color="royalblue")
    #axes[0, 0].text(1.1, 0.8, note, transform=axes[0, 0].transAxes)
    axes[1,0].text(0.1, 0.9, note, transform=axes[1, 0].transAxes)
    axes[0, 1].set_visible(False)
    fig.savefig("./N_samples.png")

    #done!
    print('Done!')

