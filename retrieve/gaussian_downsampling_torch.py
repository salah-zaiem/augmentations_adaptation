# !/usr/bin/env python
#
# Author : the Zero Resource Challenge Team
#
# In this Module, we take as input features and a
# desired number of output means, and we output
# a downsampled version of the features.

import os
import sys
import numpy as np
import pdb
from scipy.stats import norm
import torch


def padding_pca(features, padding_size) :
    element= features.ravel()
    flat_element = np.pad(element, (0, padding_size - element.shape[0]), mode = "constant", constant_values = 0 )
    #print("flat_element_shape : {}".format(fllat_element.shape))
    return flat_element

def downsample(features, n_samples, std_ratio, std_slope):
    '''
    Returns features of size n_samples * dim_features
    sampled at a regular interval with a gaussian filter;
    its standard deviation varies with the length of the input and also
    the position of the downsampling.

    Parameters
    ----------
    features : numpy.ndarray
       Array with continous features (MFCC, PLP, etc)

    n_samples: int
       the number of features to return

    std_ratio: float
        the ratio of the standard deviation with respect to the length of the input feats.

    std_slope: float
        the slope for the linear transformation of the standard deviation.

    Returns
    -------
    downsampled_feats: numpy.ndarray
        the downsampled features
    '''
    n_frames = features.shape[0]
    dim_feats = features.shape[1]

    # Create means and stds for filters
    means = np.linspace(0, n_frames-1, num=n_samples).reshape(n_samples, 1)
    ## Linear transformation for the standard deviation,
    ## according to the downsampling position.
    if n_samples % 2 == 0:
        stds = np.arange(n_samples/2)
        stds = np.append(stds, stds[::-1])
    else :
        stds = np.arange(n_samples/2 + 1)
        stds = np.append(stds, stds[:-1][::-1])
    stds = stds * std_slope + n_frames * std_ratio
    stds = stds.reshape(n_samples, 1)

    # Compute array of centered frame indices
    indices = np.arange(n_frames+1).reshape(1, n_frames+1)
    indices = np.repeat(indices - 0.5, n_samples, axis=0)

    ## One computes filters of dimension n_samples * n_frames
    ## by computing the cumulative distribution values of the previously
    ## computed indices, which then allows to compute approximate
    ## probabilities of the gaussian distribution by substracting two
    ## such consecutive values.
    cumulative_distributions = norm.cdf(indices - means, scale=stds)
    gaussian_filters = np.array(
        [cumulative_distributions[:,i+1] - cumulative_distributions[:,i]
         for i in range(n_frames)]
    ).T

    try:
        normalized_gaussian_filters =\
            gaussian_filters / gaussian_filters.sum(axis=1).reshape(n_samples, 1)
    except:
        normalized_gaussian_filters = gaussian_filters

    # Downsample features
    downsampled_features = torch.matmul(torch.tensor(normalized_gaussian_filters).cuda().float(), features)

    return downsampled_features
