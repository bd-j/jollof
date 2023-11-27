import numpy as np
import matplotlib.pyplot as plt
import corner
from tqdm import tqdm
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import norm
from nautilus import Prior
from nautilus import Sampler

# survey volume in Mpc^3
def survey_volume(area,zmin,zmax):
	#area is in arcmin^2
	omega = (area/60**2)/(4*np.pi*(180./np.pi)**2) #fraction of sky
	R_lower = cosmo.comoving_distance(zmin).value
	R_upper = cosmo.comoving_distance(zmax).value
	Vtot = omega*(4.0*np.pi/3.0)*(R_upper**3 - R_lower**3)
	return Vtot # Mpc^3

# schechter luminosity function
def schechter(Muv,phistar,mstar,alpha):
    return (0.4*np.log(10))*phistar*(10**(0.4*(mstar-Muv)))**(alpha+1) * np.exp(-10**(0.4*(mstar-Muv)))


# schechter luminosity function, averaged over a bin
def int_schechter(Muva,Muvb,logphistar,mstar,alpha,nm=100):
    Muv = np.linspace(Muva,Muvb,nm)
    phistar = 10**logphistar
    phi = schechter(Muv,phistar,mstar,alpha)
    phi_int = np.trapz(phi,x=Muv)
    return phi_int/(Muvb-Muva)

#create a mock, binned luminosity function with uncertainties
def create_mock_lf(logphistar,mstar,alpha,Volume,Muv_min=-22,Muv_max=-17,dMuv_mock=0.5):
    Muv_lower_mock = np.arange(Muv_min,Muv_max,dMuv_mock,dtype=np.float32)
    Muv_upper_mock = Muv_lower_mock+dMuv_mock
    Muv_mock = 0.5*(Muv_upper_mock+Muv_lower_mock)
    phi_mock = np.array([int_schechter(Muv_lower_mock[i],Muv_upper_mock[i],logphistar,mstar,alpha) for i in range(len(Muv_mock))])
    n_mock = phi_mock*Volume
    phi_err_mock = phi_mock/np.sqrt(n_mock)
    return Muv_lower_mock, Muv_upper_mock, Muv_mock, phi_mock, phi_err_mock

#create nautilus lf parameter priors
def create_lf_priors(lpsa=-6,lpsb=0,msa=-21,msb=-17,asa=-2.5,asb=-1.5):
	prior = Prior()
	prior.add_parameter('logphistar', dist=(lpsa, lpsb))
	prior.add_parameter('mstar', dist=(msa,msb))
	prior.add_parameter('alpha', dist=(asa,asb))
	return prior

#log likelihood
def lf_likelihood(param_dict):
    logphistar = param_dict['logphistar']
    mstar      = param_dict['mstar']
    alpha      = param_dict['alpha']
    nll = len(phi_data)
    y_ll       = np.zeros(nll)
    loglike = 0
    for i in range(nll):
        y_ll[i] = schechter(Muv_data[i],10**logphistar,mstar,alpha)
        loglike += norm.logpdf(y_ll[i],loc=phi_data[i],scale=phi_err_data[i])
    return loglike


#perform sampling
def lf_sampling(prior,likelihood,n_live=1000,verbose=False):
	sampler = Sampler(prior,likelihood,n_live=n_live)
	sampler.run(verbose=verbose)
	return sampler


#save corner plot to a file
def plot_corner(sampler,prior,fname='corner.png'):
	points, log_w, log_l = sampler.posterior()
	ndim = points.shape[1]
	fig, axes = plt.subplots(ndim, ndim, figsize=(6, 6))
	corner.corner(points, weights=np.exp(log_w), bins=20, labels=prior.keys,
              plot_datapoints=False, plot_density=False,
              fill_contours=True, levels=(0.68, 0.95),
              range=np.ones(ndim) * 0.999, fig=fig)
	plt.savefig(fname,bbox_inches='tight',dpi=400,facecolor='white')

#marginalize lf from samples
def marginalize_lf_from_samples(points,log_w,Muv_min=-22,Muv_max=-16,n_Muv=100,verbose=False):
	Muv_plot = np.linspace(Muv_min,Muv_max,n_Muv)

	phi_array = np.zeros((len(points[:,2]),n_Muv))
	for i in tqdm(range(n_Muv)):
	    phi_array[:,i] = schechter(Muv_plot[i],10**points[:,0],points[:,1],points[:,2])

	phistarm = 10**np.sum(points[:,0]*np.exp(log_w) )/np.sum(np.exp(log_w))
	mstarm =  np.sum(points[:,1]*np.exp(log_w) )/np.sum(np.exp(log_w))
	alpham = np.sum(points[:,2]*np.exp(log_w) )/np.sum(np.exp(log_w)) 
	if(verbose):
		print(f'phistar {phistarm}')
		print(f'mstar {mstarm}')
		print(f'alpha {alpham}')
	phi_plot = schechter(Muv_plot,phistarm,mstarm,alpham)
	phi_lower_plot = np.zeros_like(Muv_plot)
	phi_upper_plot = np.zeros_like(Muv_plot)

	w = np.exp(log_w)
	for i in range(n_Muv):
	    ips = np.argsort(phi_array[:,i])
	    x_phi = phi_array[ips,i].copy() #phi sorted
	    w_phi = w[ips] #w sorted by phi

	    w_int_phi = np.cumsum(w_phi) #cdf
	    w_int_phi/=w_int_phi[-1] # normalize

	    phi_lower_plot[i] = np.interp(0.16,w_int_phi,x_phi)
	    phi_upper_plot[i] = np.interp(0.84,w_int_phi,x_phi)

	return Muv_plot,phi_plot,phi_lower_plot,phi_upper_plot


#save lf plot to a file
def save_lf_plot(Muv_plot,phi_plot,phi_lower_plot,phi_upper_plot,Muv_data,phi_data,phi_err_data,Muv_lower_data,Muv_upper_data,fname='lf_mock.png',mtitle='Mock'):


	f, ax = plt.subplots(1,1,figsize=(6,6))
	ax.plot(Muv_plot,phi_plot,color='white',zorder=2)
	ax.plot(Muv_plot,phi_upper_plot,color='0.5',zorder=2)
	ax.plot(Muv_plot,phi_lower_plot,color='0.5',zorder=2)
	ax.fill_between(Muv_plot,phi_upper_plot,y2=phi_lower_plot,color='dodgerblue',alpha=0.5,zorder=1)

	ax.errorbar(Muv_data,phi_data,phi_err_data,xerr=(Muv_data-Muv_lower_data,Muv_upper_data-Muv_data),fmt='o',label='Data',zorder=4)


	ax.set_xlim([-23,-16])
	ax.set_ylim([1e-9,1.0e-1])
	ax.set_yscale('log')


	lsf = 16
	ax.set_xlabel(r'M$_{UV}$ [Absolute UV Magnitude]',fontsize=lsf)
	ax.set_ylabel(r'$\phi_{UV}$ [mag$^{-1}$ Mpc$^{-3}$]',fontsize=lsf)
	ax.legend(loc='lower right',frameon=False)
	ax.text(-22.8,4e-3,mtitle,fontsize=lsf)
	plt.savefig(fname,bbox_inches='tight',dpi=400)


#process the whole mock
def process_mocks(logphistar=-2,mstar=-19,alpha=-2,zmin=11.5,zmax=13.5,area=9.05):

	#get survey volume
	Volume = survey_volume(area,zmin,zmax)

	#create a mock lf
	Muv_lower_data, Muv_upper_data, Muv_data_mock, phi_data_mock, phi_err_data_mock = create_mock_lf(logphistar,mstar,alpha,Volume)

	global phi_data, phi_err_data, Muv_data
	phi_data = phi_data_mock
	phi_err_data = phi_err_data_mock
	Muv_data = Muv_data_mock

	#create priors
	prior = create_lf_priors()

	#perform sampling
	sampler = lf_sampling(prior,lf_likelihood,verbose=True)

	#plot the corner plot
	plot_corner(sampler,prior)

	#plot the lf
	points, log_w, log_l = sampler.posterior()
	Muv_plot,phi_plot,phi_lower_plot,phi_upper_plot = marginalize_lf_from_samples(points,log_w)

	save_lf_plot(Muv_plot,phi_plot,phi_lower_plot,phi_upper_plot,Muv_data,phi_data,phi_err_data,Muv_lower_data,Muv_upper_data)



