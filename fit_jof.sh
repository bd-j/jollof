python3  jof.py \
         --jof_datafile data/samples.v094.11172023.fits \
	 --detection data/detection_completeness.fits \
	 --selection data/selection_completeness.fits \
         --nz 100 --nl 200 --n_samples 100 \
         --evolving 0 \
	--fitter nautilus \
	--verbose 
         
#        --nz 100 --nl 200 --n_samples 10000 \
#         --nz 100 --nl 200 --n_samples 100 \
###--jof_datafile data/samples.v094.11122023.fits \
###--jof_datafile data/samples.v094.baseline_sample.fits \
###--fitter nautilus

###--fitter ultranest
