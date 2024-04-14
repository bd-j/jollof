# jollof

Inference of a parametric, evolving LF.

* `lf.py` - The main LF and completeness and effective volume classes
* `infer.py`- The likelihood, and tools for inferring parameters
* `data.py` - The data samples (in z and luminosity)

## Examples

 * `mock.py`
 * `jof.py`


## Data Formats

Jollof expects data in the form of FITS binary table with the following columns
that give samples (from the likelihood) for each object.

* `z_samples` float, shape `(n_samples,)`
* `MUV_samples`, float, shape `(n_samples,)`

Each row of the table should be a different object.  Note each entry in these
columns is a *vector* of length `n_samples`.

You can have additional columns (e.g. `"id"`)

### Completeness & Selection

Currently the selection and detection completeness functions should be provided
as FITS files containing three extensions - the 2D image with the value of the
completeness and the 1d vectors describing the axes of this image.  But
constructing the inputs to EffectiveVolumeGrid is currently the responsibility
of the user, see `jof.py` for an example.


### Sample properties


### Output


### Motivation

A rate density describes the expected (differential) number of occurrences as a
function of the N-dimensional vector ${\bf x}$.  We denote it $\rho({\bf x} |
\theta)$ where the parameters $\theta$ describe the shape of the function.  We
can then ask, given this inhomogenous Poisson process, what is the likelihood of
observing a particular set of K objects at the values $\{{\bf x}_k\}$?  This is
given by
$$p(\{{\bf x}_k\}) = e^{-N_\theta} \prod_{k=1}^K \rho({\bf x}_k | \theta)$$

where $N_\theta = \int d{\bf x} \rho({\bf x} | \theta)$.
Intuitively, you can kind of think of this as an initial 'parsimony' term that
'wants' the total expected number of occurrences to be small, working against a
second term that 'wants' to have the maximum rate density at each of the
observed values ${\bf x}_k$.

However, we rarely know ${\bf x}$ perfectly for each object, and hence do not
know $\gamma_k=\rho({\bf x}_k | \theta)$.  Instead, we have some noisy estimate
or likelihood for ${\bf x}_k$, which we will call $p_k({\bf x}_k)$. But, we can
still compute the expected rate $\gamma$ for the $k$th object if we
*marginalize* over the true value ${\bf x}_k$ by integrating
$\gamma=\int d{\bf x}_k p_k({\bf x}_k) \rho({\bf x}_k | \theta)$.
It is often convenient to do this integral numerically using $J$ fair samples of
${\bf x}_k$ from the distribution $p_k({\bf x}_k)$:
$$\gamma_k = \int d{\bf x}_k p_k({\bf x}_k) \rho({\bf x}_k | \theta) \sim \sum_j \rho({\bf x}_{k,j} | \theta)$$

This marginalization correctly accounts for the uncertainties on each object via
a forward model, resulting in a probability for the *error-deconvolved* rate
density $\rho$. Putting this all together, we get
$$p(\{{\bf x}_k\}) = e^{-N_\theta}  \prod_{k=1}^K \sum_j \rho({\bf x}_{k,j} | \theta)$$


I skipped over an important subtlety in the above.  We cannot use just *any*
distribution for $p_k({\bf x}_k)$. In the marginalization we did, we effectively
used $\rho$ as a *prior* on ${\bf x}$ (and we are allowing that prior to change
if we change $\theta$).  This suggests that we want to make sure we have already
removed any prior on ${\bf x}$ that enters in to $p_k$ and are using instead a
*likelihood* of the data given ${\bf x}_k$.  Furthermore, because we are
constructing the *product* of all the $\gamma$s, any prior that is already there
will be raised to the $K$th power.

Removing a prior can be done in a number of ways, but one of them is to divide by the prior probability during the marginalization integral (either analytic or numeric).