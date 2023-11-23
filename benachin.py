#import sys
#import os
import numpy as np
#import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
#from astropy.wcs import WCS
import argparse
from tqdm import tqdm
import time
from astropy.cosmology import Planck15 as cosmo
import lf
from astropy.units import arcmin
from scipy.stats import rv_discrete

def DistanceModulus(z):

    dl = 1.0e6*cosmo.luminosity_distance(z).value #lum distance to z in pc
    K_correction = -2.5*np.log10(1+z) #K correction, see Hogg 1999, eq 27

    return 5.0*(np.log10(dl)-1.0) + K_correction

#######################################
# Create command line argument parser
#######################################

def create_parser():

	# Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="Fake LF counts flags and options from user.")

    parser.add_argument('-o', '--output',
        dest='output',
        default='output.fits',
        metavar='output',
        type=str,
        help='Output file.')

    parser.add_argument('-a', '--area',
        dest='area',
        default=9.05,
        metavar='area',
        type=float,
        help='Area in arcmin.')

 #Minimum redshift
    parser.add_argument('--zmin',
        default=11.75,
        metavar='zmin',
        type=float,
        help='Minimum redshift to model (default: 12)')

    #Maximum redshift
    parser.add_argument('--zmax',
        default=12.25,
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
        default=17.5/2.5,
        metavar='loglmin',
        type=float,
        help=f'Minimum log luminosity (absolute maggies) to model (default: {17.5/2.5})')

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

    #Luminosity samples
    parser.add_argument('--n_samples',
        default=1000,
        metavar='n_samples',
        type=int,
        help='Number of MUv, z samples')


    #Evovling luminosity function parameters
    parser.add_argument('--lf_params',
        nargs=5,
        default=[-4., 0, (19 / 2.5), 0, -2.0],
#        default=[-2., 0, (18 / 2.5), 0, -1.8],
#        default=[-3., 0, (19 / 2.5), 0, -2.0],

        type=float,
        help='LF parameters')

    #Complete
    parser.add_argument('--complete',
        dest='complete',
        action='store_true',
        help='Model LF as complete? (default: False)',
        default=False)


    parser.add_argument('-v', '--verbose',
        dest='verbose',
        action='store_true',
        help='Print helpful information to the screen? (default: False)',
        default=False)

    return parser

#######################################
# main() function
#######################################
def main():

    #begin timer
    time_global_start = time.time()

    #create the command line argument parser
    parser = create_parser()

    #store the command line arguments
    args   = parser.parse_args()

    # grid of redshifts
    zgrid = np.linspace(args.zmin, args.zmax, args.nz)

    # grid of log luminosity
    loglgrid = np.linspace(args.loglmin, args.loglmax, args.nl)

    #compute covering angle
    omega = (args.area * arcmin**2).to("steradian").value

    # initialize evolving schechter
    q_true = np.array(args.lf_params)
    s = lf.EvolvingSchechter()
    s.set_parameters(q_true)

    print(f'n {s.nM(nMx=100)}')

    print(f'n {s.nl(lmin=(18./2.5),lmax=(23./2.5),nlx=100)}')

    print(f'rho {s.rhol(lmin=(18./2.5),lmax=(23./2.5),nlx=100)}')


    # ------------------------------------
    # ---- compute effective volume    ---
    # ------------------------------------
    completeness_kwargs = {'flag_complete': args.complete}
    veff = lf.construct_effective_volume(loglgrid, zgrid, omega, completeness_kwargs,\
                                      as_interpolator=True, fake_flag=True,\
                                      muv_min=25,muv_max=32, f_cover=1.)

    #save a figure
    lf.plot_veff(loglgrid,zgrid,veff)

    # this takes most of the time:
    #lf = self.evaluate(veff.lgrid, veff.zgrid, grid=True, in_dlogl=True)
    #dV_dz = veff.data  # shape (nL, nz)
    # TODO: use dot products here for the integral
    #dV = dV_dz * veff.dz[None, :]
    #dN = lf * dV * veff.dlogl[:, None]
    #return dN, dV, lf

    dN, dV = s.n_effective(veff)

    print(f'loglgrid.shape {loglgrid.shape}')

    print(f'zgrid.shape {zgrid.shape}')
    print(f'dV.shape {dV.shape}')
    #print(f'dV.sum() {dV[-1,:].sum()}')
    #print(f'dV.sum() {dV[0,:].sum()}')
    print(f'veff.dlogl[:, None].shape {veff.dlogl[:, None].shape}')
    print(f'dN.shape {dN.shape}')

    #shape is (loglgrid.shape,zgrid.shape)
    N_bar = dN.sum()  # Average number of galaxies in survey
    V_bar = dV[-1, :].sum()  # Effective volume of the survey in Mpc^3


    import matplotlib.pyplot as plt
    plt.clf()
    plt.imsave('dV.png',dV)


    if (args.verbose):
        print(f'Area in arcmin^2 = {args.area}')
        print(f'Area in steraidians = {omega}')
        print(f'Minimum redshift = {args.zmin}')
        print(f'Maximum redshift = {args.zmax}')
        print(f'Cosmology: h = {cosmo.h}, Omega_m = {cosmo.Om0}')
        print(f'Expected number: {N_bar}.')
        print(f'TESTING')
        print(f'nl * V_bar {s.nl(q=q_true,lmin=7.0,lmax=25.)*V_bar}')
        print(f'Effective volume = {V_bar} [Mpc^3].')
        print(f'Number density = {N_bar/V_bar} [Mpc^-3]')
        print(f'Number density (analytical) = {s.nl(q=q_true,lmin=7.0,lmax=25.)} [Mpc^-3]')
        print(f'Luminosity density (analytical, MUV<-18) = {s.rhol(lmin=7.2,lmax=25.)/1e25} [10^25 erg s^-1 Hz^-1 Mpc^-3]')
        print(f'Luminosity density (analytical, Total)   = {s.rhol(lmin=-10,lmax=25.)/1e25} [10^25 erg s^-1 Hz^-1 Mpc^-3]')

    #pull random variates from luminosity function

    #first, construct N(z), which sets which redshift
    #each object should be

    nz = np.zeros_like(zgrid)
    for i in range(len(zgrid)):
        nz[i] = s.nl(z=zgrid[i],q=q_true,lmin=args.loglmin,lmax=args.loglmax)
    nz*=np.gradient(zgrid)*dV[-1,:] #account for volume variation with redshift
#    nz*=np.gradient(zgrid) #account for volume variation with redshift

    nz/=nz.sum()

    #pull zgrid samples from n(z) distribution 
    rng_nz = rv_discrete(name='z_samples',values=(zgrid,nz))

    #number of objects
    nobj = int(N_bar)

    #redshift locations
    z_samples = rng_nz.rvs(size=nobj)
    #z_samples.sort()

    print(z_samples)

    #ok, for each object, we need to pull from the
    #luminosity function at that redshift
    iz = np.searchsorted(zgrid,z_samples)

    #print(zgrid[iz])
    #print(dN[:,iz[0]])

    l_samples = np.zeros_like(z_samples)
    for k in range(nobj):
        i = iz[k]
        dN_z = dN[:,i].copy()
        dN_z/=dN_z.sum()
        rng_nz = rv_discrete(name='l_sample',values=(loglgrid,dN_z))
        l_samples[k] = rng_nz.rvs(size=1)[0]

    print(l_samples)

    dm = DistanceModulus(z_samples)
    mab = -2.5*l_samples + dm

    MUV_samples = np.zeros((nobj,args.n_samples))
    z_samples = np.zeros((nobj,args.n_samples))

    print(f'Pulling {args.n_samples} samples for {nobj} objects....')
    for j in tqdm(range(nobj)):
        z_samples[j,:] =    np.random.uniform(low=args.zmin,high=args.zmax,size=args.n_samples)
        MUV_samples[j,:] = mab[j] - DistanceModulus(z_samples[j,:])


    #save as a fits file

    #copy the photometric catalog info for these objects\
    #to fits hdus
    hdu_primary = fits.PrimaryHDU()
    hdus = [hdu_primary]

    hdu_tmp = fits.HDUList(hdus=hdus)

    col1 = fits.Column(name='z_samples',format=f'{args.n_samples}E')
    col2 = fits.Column(name='MUV_samples',format=f'{args.n_samples}E')
    coldefs = fits.ColDefs([col1,col2])
    hdu_zs = fits.BinTableHDU.from_columns(coldefs,nrows=nobj,name='ZSAMP')
    hdu_zs.data['z_samples'] = z_samples
    hdu_zs.data['MUV_samples'] = MUV_samples
    hdus.append(hdu_zs)

    #output the results
    hdu_out = fits.HDUList(hdus=hdus)
    hdu_out.writeto(args.output,overwrite=True)

    #end timer
    time_global_end = time.time()
    if(args.verbose):
    	print(f"Time to execute program: {time_global_end-time_global_start}s.")

#######################################
# Run the program
#######################################
if __name__=="__main__":
	main()