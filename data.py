import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits
from astropy.table import Table

# ------------------------
# Data storage (and mocks)
# ------------------------
class DataSamples:
    """Holds posterior samples for logL and zred for all objects being used to
    constrain the LF.  Optionally include flux samples to incorporate
    k-correction effects or other selection effects.  Could also include radius
    samples.
    """

    def __init__(self, objects=None, filename=None, ext="SAMPLES", n_samples=1000, replicate=0):

        self.all_samples = None
        self.n_samples = n_samples
        if filename is not None:
            self.all_samples = self.rectify_eazy(filename, ext, replicate=replicate)
            self.n_samples = len(self.all_samples[0]["zred_samples"])
        if objects is not None:
            self.add_objects(objects)

    def rectify_eazy(self, filename, ext, replicate=0):
        table = Table.read(filename, hdu=ext)
        convert = dict(zred_samples=("z_samples", lambda x: x),
                       logl_samples=("MUV_samples", lambda x: -0.4 * x))

        dtype = self.get_dtype(self.n_samples)
        all_samples = np.zeros(len(table), dtype=dtype)
        for n, (o, konvert) in convert.items():
            all_samples[:][n] = konvert(table[o][:, :self.n_samples])

        #Add duplicate objects (for testing purposes)
        if(replicate>0):
            print(f'Replicating the sample {replicate} times.')
            print(f'all_samples.shape {all_samples.shape} all_samples.keys() {all_samples[0].dtype}')
            new_samples = all_samples.copy()
            for i in range(replicate):
                new_samples = np.append(new_samples,all_samples)
            all_samples = new_samples
            print(f'all_samples.shape {all_samples.shape} all_samples.keys() {all_samples[0].dtype}')

        return all_samples

    def to_eazy(self):
        convert = dict(z_samples=("zred_samples", lambda x: x),
                       MUV_samples=("logl_samples", lambda x: -2.5 * x))
        n_samples = self.n_samples
        dtype = np.dtype([("MUV_samples", float, (n_samples,)),
                          ("z_samples", float, (n_samples,))])
        arr = np.zeros(len(self.all_samples), dtype=dtype)
        for n, (o, konvert) in convert.items():
            arr[:][n] = konvert(self.all_samples[o][:, :self.n_samples])
        return arr

    def add_objects(self, objects):
        dtype = self.get_dtype(self.n_samples)
        new = np.zeros(len(objects), dtype=dtype)
        for i, obj in enumerate(objects):
            for k in obj.keys():
                new[i][k] = obj[k]
        if self.all_samples is None:
            self.all_samples = new
        else:
            self.all_samples = np.concatenate([self.all_samples, new])

    def get_dtype(self, n_samples):
        truecols = [("logl_true", float), ("zred_true", float)]
        sample_cols = [("logl_samples", float, (n_samples,)),
                       ("zred_samples", float, (n_samples,)),
                       ("sample_weight", float, (n_samples,))]
        extras =  ["flux", "rhalf"] # + ["sersic", "q"]
        extra_cols = [(f"{c}_samples", float, (n_samples,)) for c in extras]
        return np.dtype([("id", int)] + sample_cols + extra_cols + truecols)

    def show(self, n_s=15, ax=None, **plot_kwargs):
        if ax is None:
            fig, ax = pl.subplots()
        else:
            fig = None
        ax.plot(self.all_samples["zred_true"], self.all_samples["logl_true"], "ro",
                zorder=20)
        for d in self.all_samples:
            ax.plot(d["zred_samples"][:n_s], d["logl_samples"][:n_s],
                    marker=".", linestyle="", color='gray')
        if fig is not None:
            fig.savefig("mock_samples.png")
        return fig, ax

    def to_fits(self, fitsfilename):
        arr = self.to_eazy()
        samples = fits.BinTableHDU(arr, name="SAMPLES")
        samples.header["NSAMPL"] = self.n_samples
        hdul = fits.HDUList([fits.PrimaryHDU(),
                             samples])
        hdul.writeto(fitsfilename, overwrite=True)