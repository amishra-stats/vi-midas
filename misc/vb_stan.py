from collections import OrderedDict
import numpy as np
from scipy.special import comb, loggamma




def vb_extract_sample(results):
    param_specs = results['sampler_param_names']
    samples = results['sampler_params']
    n = len(samples[0])

    # first pass, calculate the shape
    param_shapes = OrderedDict()
    for param_spec in param_specs:
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            idxs = [int(i) for i in splt[1][:-1].split(',')]  # no +1 for shape calculation because pystan already returns 1-based indexes for vb!
        else:
            idxs = ()
        param_shapes[name] = np.maximum(idxs, param_shapes.get(name, idxs))

    # create arrays
    params = OrderedDict([(name, np.nan * np.empty((n, ) + tuple(shape))) for name, shape in param_shapes.items()])

    # second pass, set arrays
    for param_spec, param_samples in zip(param_specs, samples):
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            idxs = [int(i) - 1 for i in splt[1][:-1].split(',')]  # -1 because pystan returns 1-based indexes for vb!
        else:
            idxs = ()
        params[name][(..., ) + tuple(idxs)] = param_samples

    return params



def vb_extract_mean(results):
    param_specs = results['mean_par_names']
    mean_est = results['mean_pars']
    
    # first pass, calculate the shape
    param_shapes = OrderedDict()
    for param_spec in param_specs:
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            idxs = [int(i) for i in splt[1][:-1].split(',')]  # no +1 for shape calculation because pystan already returns 1-based indexes for vb!
        else:
            idxs = (1,)
        param_shapes[name] = np.maximum(idxs, param_shapes.get(name, idxs))
    
    # create arrays
    params = OrderedDict([(name, np.nan * np.empty(shape)) for name, shape in param_shapes.items()])
    # second pass, set arrays
    for param_spec, est in zip(param_specs, mean_est):
        #param_spec = param_specs[500]
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            idxs = [int(i) - 1 for i in splt[1][:-1].split(',')]  # -1 because pystan returns 1-based indexes for vb!
        else:
            idxs = (0,)
        params[name][tuple(idxs)] = est
        
    return params


def convert_params(mu, phi):
    """ 
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    Parameters
    ----------
    mu : float 
       Mean of NB distribution.
    alpha : float
       Overdispersion parameter used for variance calculation.

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """
    p = mu / (mu + phi)
    r = phi
    return r, p



def neg_binomial_2_rng(mu, phi):
    """Generate a sample from neg_binomial_2(mu, phi).

    This function is defined here in Python rather than in Stan because Stan
    reports overflow errors.

    $E(Y) = mu$
    $Var(Y) = mu + mu^2 / phi

    This function will work fine with arrays.
    """
    tem = np.random.gamma(phi, mu / phi)
    if(tem > 1e15):
        return -1
    else:
        return np.random.poisson(tem)



def neg_binomial_2_lpmf2(y, mu, phi):
    '''
    Compute negative binomial log probability mass function 
    '''
    return np.log(comb(y+phi-1,y)) + y*np.log(mu/(mu+phi)) + phi*np.log(phi/(mu+phi))


def neg_binomial_2_lpmf(y, mu, phi):
    '''
    Compute negative binomial log probability mass function 
    '''
    r, p = convert_params(mu, phi)
    
    return loggamma(r+y) - loggamma(y+1) - loggamma(r) + y*np.log(p) + r*np.log(1-p) 
    
    #return np.log(comb(y+phi-1,y)) + y*np.log(mu/(mu+phi)) + phi*np.log(phi/(mu+phi))

    
    
