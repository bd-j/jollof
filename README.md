```
python3 jof.py
         --jof_datafile data/samples.v094.baseline_sample.fits \
         --nz 100 --nl 200 --n_samples 100 \
         --evolving 0 \
         --fitter nautilus
```

line 54 veff

lf = evolving schechter on line 60

reference redshift is 14 for phi0
phi1 is evolution in logphi = phi0 + phi*(z-zref)

set evolving = 0 -- no evolution


prior = Prior() line 105

uniform for phi0 between -5 and -3
phi 1 is uniform
lstar0 uniform in log lstar

Prior changes


line 85 logliki

line 112

log weigth and loeg like and points

Average == points*exp(log_w)/sum(exp_log_w)