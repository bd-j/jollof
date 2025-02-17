#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""priors.py -- This module contains various objects to be used as priors.
When called these return the ln-prior-probability, and they can also be used to
construct prior transforms (for nested sampling) and can be sampled from.
"""
import numpy as np
import scipy.stats
from scipy.special import erf, erfinv

__all__ = ["Parameters",
           "Prior", "Uniform", "TopHat", "Normal", "ClippedNormal",
           "LogNormal", "LogUniform", "Beta",
           "StudentT", "SkewNormal",
           "FastUniform", "FastTruncatedNormal",
           "FastTruncatedEvenStudentTFreeDeg2",
           "FastTruncatedEvenStudentTFreeDeg2Scalar"]


class Parameters:
    """
    This is the base model class that holds model parameters and information
    about them (e.g. priors, bounds, transforms, free vs fixed state).  In
    addition to the documented methods, it contains several important
    attributes:

    * :py:attr:`params`: model parameter state dictionary.
    * :py:attr:`theta_index`: A dictionary that maps parameter names to indices (or rather
      slices) of the parameter vector ``theta``.
    * :py:attr:`config_dict`: Information about each parameter as a dictionary keyed by
      parameter name for easy access.
    * :py:attr:`config_list`: Information about each parameter stored as a list.

    Intitialization is via, e.g.,

    .. code-block:: python

       model_dict = {"mass": {"N": 1, "isfree": False, "init": 1e10}}
       model = ProspectorParams(model_dict, param_order=None)

    :param configuration:
        A list or dictionary of model parameters specifications.
    """

    def __init__(self, param_names, priors):
        """
        :param configuration:
            A list or dictionary of parameter specification dictionaries.

        :param param_order: (optional, default: None)
            If given and `configuration` is a dictionary, this will specify the
            order in which the parameters appear in the theta vector.  Iterable
            of strings.
        """
        self.param_names = param_names
        self.free_params = param_names
        self.priors = priors
        self.theta_index = {p:[i] for i, p in enumerate(param_names)}


    def prior_product(self, theta, nested=False, **extras):
        """Public version of _prior_product to be overridden by subclasses.

        :param theta:
            The parameter vector for which you want to calculate the
            prior. ndarray of shape ``(..., ndim)``

        :param nested:
            If using nested sampling, this will only return 0 (or -inf).  This
            behavior can be overridden if you want to include complicated
            priors that are not included in the unit prior cube based proposals
            (e.g. something that is difficult to transform from the unit cube.)

        :returns lnp_prior:
            The natural log of the prior probability at ``theta``
        """
        lpp = self.prior_lnp(theta)
        if nested & np.any(np.isfinite(lpp)):
            return 0.0
        return lpp.sum()

    def prior_lnp(self, theta, **extras):
        """Return a scalar which is the ln of the product of the prior
        probabilities for each element of theta.  Requires that the prior
        functions are defined in the theta descriptor.

        :param theta:
            Iterable containing the free model parameter values. ndarray of
            shape ``(ndim,)``

        :returns lnp_prior:
            The natural log of the product of the prior probabilities for these
            parameter values.
        """
        lnp_prior = 0
        lnp_prior = np.zeros(len(self.free_params))
        for k, inds in list(self.theta_index.items()):

            func = self.priors[k]
            this_prior = np.sum(func(theta[..., inds]), axis=-1)
            lnp_prior[inds] += this_prior

        return lnp_prior

    def prior_transform(self, unit_coords):
        """Go from unit cube to parameter space, for nested sampling.

        :param unit_coords:
            Coordinates in the unit hyper-cube. ndarray of shape ``(ndim,)``.

        :returns theta:
            The parameter vector corresponding to the location in prior CDF
            corresponding to ``unit_coords``. ndarray of shape ``(ndim,)``
        """
        theta = np.zeros(len(unit_coords))
        for k, inds in list(self.theta_index.items()):
            func = self.priors[k].unit_transform
            theta[inds] = func(unit_coords[inds])
        return theta


class Prior(object):
    """Encapsulate the priors in an object.  Each prior should have a
    distribution name and optional parameters specifying scale and location
    (e.g. min/max or mean/sigma).  These can be aliased at instantiation using
    the ``parnames`` keyword. When called, the argument should be a variable
    and the object should return the ln-prior-probability of that value.

    .. code-block:: python

        ln_prior_prob = Prior(param=par)(value)

    Should be able to sample from the prior, and to get the gradient of the
    prior at any variable value.  Methods should also be avilable to give a
    useful plotting range and, if there are bounds, to return them.

    Parameters
    ----------
    parnames : sequence of strings
        A list of names of the parameters, used to alias the intrinsic
        parameter names.  This way different instances of the same Prior can
        have different parameter names, in case they are being fit for....

    Attributes
    ----------
    params : dictionary
        The values of the parameters describing the prior distribution.
    """

    def __init__(self, parnames=[], name='', **kwargs):
        """Constructor.

        Parameters
        ----------
        parnames : sequence of strings
            A list of names of the parameters, used to alias the intrinsic
            parameter names.  This way different instances of the same Prior
            can have different parameter names, in case they are being fit for....
        """
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)
        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name
        self.update(**kwargs)

    def __repr__(self):
        argstring = ['{}={}'.format(k, v) for k, v in list(self.params.items())]
        return '{}({})'.format(self.__class__, ",".join(argstring))

    def update(self, **kwargs):
        """Update ``self.params`` values using alias.
        """
        for k in self.prior_params:
            try:
                self.params[k] = kwargs[self.alias[k]]
            except(KeyError):
                pass
        # FIXME: Should add a check for unexpected kwargs.

    def __len__(self):
        """The length is set by the maximum size of any of the prior_params.
        Note that the prior params must therefore be scalar of same length as
        the maximum size of any of the parameters.  This is not checked.
        """
        return max([np.size(self.params.get(k, 1)) for k in self.prior_params])

    def __call__(self, x, **kwargs):
        """Compute the value of the probability desnity function at x and
        return the ln of that.

        Parameters
        ----------
        x : float or sequqnce of float
            Value of the parameter, scalar or iterable of same length as the
            Prior object.

        kwargs : optional
            All extra keyword arguments are used to update the `prior_params`.

        Returns
        -------
        lnp : float or sequqnce of float, same shape as ``x``
            The natural log of the prior probability at ``x``, scalar or ndarray
            of same length as the prior object.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        pdf = self.distribution.pdf
        try:
            p = pdf(x, *self.args, loc=self.loc, scale=self.scale)
        except(ValueError):
            # Deal with `x` vectors of shape (nsamples, len(prior))
            # for pdfs that don't broadcast nicely.
            p = [pdf(_x, *self.args, loc=self.loc, scale=self.scale)
                 for _x in x]
            p = np.array(p)

        with np.errstate(invalid='ignore'):
            lnp = np.log(p)
        return lnp

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.

        :param nsample: (optional)
            Unused
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.rvs(*self.args, size=len(self),
                                     loc=self.loc, scale=self.scale)

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.

        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.

        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.ppf(x, *self.args,
                                     loc=self.loc, scale=self.scale)

    def inverse_unit_transform(self, x, **kwargs):
        """Go from the parameter value to the unit coordinate using the cdf.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.cdf(x, *self.args,
                                     loc=self.loc, scale=self.scale)

    def gradient(self, theta):
        raise(NotImplementedError)

    @property
    def loc(self):
        """This should be overridden.
        """
        return 0

    @property
    def scale(self):
        """This should be overridden.
        """
        return 1

    @property
    def args(self):
        return []

    @property
    def range(self):
        raise(NotImplementedError)

    @property
    def bounds(self):
        raise(NotImplementedError)

    def serialize(self):
        raise(NotImplementedError)


class Uniform(Prior):
    """A simple uniform prior, described by two parameters

    :param mini:
        Minimum of the distribution

    :param maxi:
        Maximum of the distribution
    """
    prior_params = ['mini', 'maxi']
    distribution = scipy.stats.uniform

    @property
    def kind(self):
        return "Uniform"

    @property
    def scale(self):
        return self.params['maxi'] - self.params['mini']

    @property
    def loc(self):
        return self.params['mini']

    @property
    def range(self):
        return (self.params['mini'], self.params['maxi'])

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class TopHat(Uniform):
    """Uniform distribution between two bounds, renamed for backwards compatibility
    :param mini:
        Minimum of the distribution

    :param maxi:
        Maximum of the distribution
    """


class Normal(Prior):
    """A simple gaussian prior.


    :param mean:
        Mean of the distribution

    :param sigma:
        Standard deviation of the distribution
    """
    prior_params = ['mean', 'sigma']
    distribution = scipy.stats.norm

    @property
    def kind(self):
        return "Normal"

    @property
    def scale(self):
        return self.params['sigma']

    @property
    def loc(self):
        return self.params['mean']

    @property
    def range(self):
        nsig = 4
        return (self.params['mean'] - nsig * self.params['sigma'],
                self.params['mean'] + nsig * self.params['sigma'])

    def bounds(self, **kwargs):
        #if len(kwargs) > 0:
        #    self.update(**kwargs)
        return (-np.inf, np.inf)


class ClippedNormal(Prior):
    """A Gaussian prior clipped to some range.

    :param mean:
        Mean of the normal distribution

    :param sigma:
        Standard deviation of the normal distribution

    :param mini:
        Minimum of the distribution

    :param maxi:
        Maximum of the distribution
    """
    prior_params = ['mean', 'sigma', 'mini', 'maxi']
    distribution = scipy.stats.truncnorm

    @property
    def kind(self):
        return "ClippedNormal"

    @property
    def scale(self):
        return self.params['sigma']

    @property
    def loc(self):
        return self.params['mean']

    @property
    def range(self):
        return (self.params['mini'], self.params['maxi'])

    @property
    def args(self):
        a = (self.params['mini'] - self.params['mean']) / self.params['sigma']
        b = (self.params['maxi'] - self.params['mean']) / self.params['sigma']
        return [a, b]

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class LogUniform(Prior):
    """Like log-normal, but the distribution of natural log of the variable is
    distributed uniformly instead of normally.

    :param mini:
        Minimum of the distribution

    :param maxi:
        Maximum of the distribution
    """
    prior_params = ['mini', 'maxi']
    distribution = scipy.stats.reciprocal

    @property
    def kind(self):
        return "LogUniform"

    @property
    def args(self):
        a = self.params['mini']
        b = self.params['maxi']
        return [a, b]

    @property
    def range(self):
        return (self.params['mini'], self.params['maxi'])

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class Beta(Prior):
    """A Beta distribution.

    :param mini:
        Minimum of the distribution

    :param maxi:
        Maximum of the distribution

    :param alpha:

    :param beta:
    """
    prior_params = ['mini', 'maxi', 'alpha', 'beta']
    distribution = scipy.stats.beta

    @property
    def kind(self):
        return "Beta"

    @property
    def scale(self):
        return self.params.get('maxi', 1) - self.params.get('mini', 0)

    @property
    def loc(self):
        return self.params.get('mini', 0)

    @property
    def args(self):
        a = self.params['alpha']
        b = self.params['beta']
        return [a, b]

    @property
    def range(self):
        return (self.params.get('mini',0), self.params.get('maxi',1))

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class LogNormal(Prior):
    """A log-normal prior, where the natural log of the variable is distributed
    normally.  Useful for parameters that cannot be less than zero.

    Note that ``LogNormal(np.exp(mode) / f) == LogNormal(np.exp(mode) * f)``
    and ``f = np.exp(sigma)`` corresponds to "one sigma" from the peak.

    :param mode:
        Natural log of the variable value at which the probability density is
        highest.

    :param sigma:
        Standard deviation of the distribution of the natural log of the
        variable.
    """
    prior_params = ['mode', 'sigma']
    distribution = scipy.stats.lognorm

    @property
    def kind(self):
        return "LogNormal"

    @property
    def args(self):
        return [self.params["sigma"]]

    @property
    def scale(self):
        return  np.exp(self.params["mode"] + self.params["sigma"]**2)

    @property
    def loc(self):
        return 0

    @property
    def range(self):
        nsig = 4
        return (np.exp(self.params['mode'] + (nsig * self.params['sigma'])),
                np.exp(self.params['mode'] - (nsig * self.params['sigma'])))

    def bounds(self, **kwargs):
        return (0, np.inf)


class StudentT(Prior):
    """A Student's T distribution

    :param mean:
        Mean of the distribution

    :param scale:
        Size of the distribution, analogous to the standard deviation

    :param df:
        Number of degrees of freedom
    """
    prior_params = ['mean', 'scale', 'df']
    distribution = scipy.stats.t

    @property
    def kind(self):
        return "StudentT"

    @property
    def args(self):
        return [self.params['df']]

    @property
    def scale(self):
        return self.params['scale']

    @property
    def loc(self):
        return self.params['mean']

    @property
    def range(self):
        return scipy.stats.t.interval(0.995, self.params['df'], self.params['mean'], self.params['scale'])

    def bounds(self, **kwargs):
        return (-np.inf, np.inf)
