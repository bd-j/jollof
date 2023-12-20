import numpy as np
import matplotlib.pyplot as plt


# ------------------------------
# Plot detection completeness grid
# ------------------------------
def plot_detection(mab,lrh,comp):
    f,ax = plt.subplots(1,1,figsize=(6,6))
    dmab = mab[1]-mab[0]
    dlrh = lrh[1]-lrh[0]
    x_min = mab[0]-0.5*dmab
    x_max = mab[-1]+0.5*dmab
    y_min = lrh[0]-0.5*dlrh
    y_max = lrh[-1]+0.5*dlrh

    #im = ax.imshow(comp.T,origin='lower',extent= \
    x,y = np.meshgrid(mab,lrh)
    im = ax.imshow(comp((x,y)),origin='lower',extent= \
        [x_min,x_max,y_min,y_max])
    ax.set_xlabel('Apparent Magnitude [AB]')
    ax.set_ylabel('Log10 Rhalf [arcsec]')
    ax.set_xlim([x_min,x_max])
    ax.set_ylim([y_min,y_max])
    ax.set_aspect((x_max-x_min)/(y_max-y_min))
    cb = f.colorbar(im,ax=ax,label='Completeness',fraction=0.046, pad=0.04)
    print('Writing detection_completeness.png.')
    plt.savefig('detection_completeness.png',bbox_inches='tight',dpi=400)

# ------------------------------
# Plot selection completeness grid
# ------------------------------
def plot_selection(z,muv,comp):
    f,ax = plt.subplots(1,1,figsize=(6,6))
    dx = z[1]-z[0]
    dy = muv[1]-muv[0]
    x_min = z[0]-0.5*dx
    x_max = z[-1]+0.5*dx
    y_min = muv[0]-0.5*dy
    y_max = muv[-1]+0.5*dy

#    im = ax.imshow(comp.T,origin='lower',extent= \
    x,y = np.meshgrid(z,muv)
    im = ax.imshow(comp((x,y)),origin='lower',extent= \
        [x_min,x_max,y_min,y_max])
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel('Absolute MUV')
    ax.set_xlim([x_min,x_max])
    ax.set_ylim([y_min,y_max])
    ax.set_aspect((x_max-x_min)/(y_max-y_min))
    cb = f.colorbar(im,ax=ax,label='Completeness',fraction=0.046, pad=0.04)
    print('Writing selection_completeness.png.')
    plt.savefig('selection_completeness.png',bbox_inches='tight',dpi=400)

# ------------------------------
# Plot effective volume
# ------------------------------
def plot_veff(loglgrid,zgrid,veff):
    f,ax = plt.subplots(1,1,figsize=(6,6))
    dx = zgrid[1]-zgrid[0]
    dy = loglgrid[1]-loglgrid[0]
    x_min = zgrid[0]-0.5*dx
    x_max = zgrid[-1]+0.5*dx
    y_min = loglgrid[0]-0.5*dy
    y_max = loglgrid[-1]+0.5*dy

#    im = ax.imshow(comp.T,origin='lower',extent= \
    x,y = np.meshgrid(zgrid,loglgrid)
    veffi = veff((y,x))
    print(f'Maximum veff {np.max(veffi)}')
    im = ax.imshow(veffi,origin='lower',extent= \
        [x_min,x_max,y_min,y_max])
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel('Log L')
    ax.set_xlim([x_min,x_max])
    ax.set_ylim([y_min,y_max])
    ax.set_aspect((x_max-x_min)/(y_max-y_min))
    cb = f.colorbar(im,ax=ax,label='Effective Volume',fraction=0.046, pad=0.04)
    print('Writing effective_volume.png.')
    plt.savefig('effective_volume.png',bbox_inches='tight',dpi=400)
