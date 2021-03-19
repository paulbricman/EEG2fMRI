# EEG2fMRI
Paul Bricman, Jelmer Borst

## Tackled Problem
Current neuroimaging techniques are limited. EEG (based on electrodes placed on scalp) has high temporal resolution (sampling frequency), yet poor spatial resolution (due to the signal being obstructed and filtered by the brain and the skull). In contrast, fMRI has high spatial resolution (due to its reliance on electromagnetic waves which penetrate the brain), yet poor temporal resolution. However, both EEG and fMRI are used to record the same thing: neural activity. It's plausible that EEG signal could be used to somehow "triangulate" neural activity at the precise locations captured by fMRI. Multi-modal neuroimaging studies provide a unique opportunity for exploring this mapping. Those studies are based on simultaneous recordings with both EEG and fMRI, providing a perfectly aligned cross-modal signal which a machine learning model could use for training. A model capable of supporting this mapping would greatly augment the spatial resolution which can be achieved with EEG. 

A metaphor might be useful for understanding the problem being tackled. Neural activity is akin to a subject being photographed. In this analogy, EEG is like an action camera with extremely poor resolution, yet which manages to record hundreds of frames per second. Conversely, fMRI is like a vintage film-based camera. It takes a long for the subject figure to get imprinted on the film, and recording multiple frames is very tedious, yet the resolution of the individual photograph is superb. Training a model to map EEG to fMRI data is like training a model to map the action camera recording onto the sequence of film-based photographs.

## Implementation

We found a dataset from a multi-modal neuroimaging study from which we derived samples with the following structure. The input data consists of a 30-second-long EEG recording across 34 electrodes at 1000Hz. This data resembles 34 parallel time series unfolding across 30 seconds. Each time series comes from a certain electrode placed in a certain place on the subject's scalp. The output data consists of a three-dimensional tensor of shape 53x63x52 which depicts the 3D "picture" of the subject's brain obtained through fMRI.

We're training on data from 15 subjects, and testing the quality of the learned mapping on a 16th subject which has been left out during training. We trained models with fully-connected, convolutional, transformer-based, and mixed architectures. We tweaked activation functions, loss functions, optimizers, learning rates, batch size, layer sizes, learning rate schedules, and others.

