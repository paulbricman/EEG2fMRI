from torch.utils.data import Dataset
import torch

import nibabel as nib
from scipy.io import loadmat

import os
import numpy as np
import matplotlib.pyplot as plt


class OddballDataset(Dataset):
    """Auditory and Visual Oddball EEG-fMRI dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Path to the root folder containing subject subfolders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform

        # Setting constants based on scanner specifications
        self.samples_per_block = 155
        self.blocks_per_subject = 6
        self.samples_per_subject = self.samples_per_block * self.blocks_per_subject
        self.subject_count = 16
        self.skipped_samples_per_block = 170 - self.samples_per_block
        self.eeg_frequency = 1000  # Hz
        self.fmri_period = 2  # seconds
        self.skipped_eeg_per_block = self.skipped_samples_per_block * \
            self.fmri_period * self.eeg_frequency
        self.eeg_sample_step = self.fmri_period * self.eeg_frequency
        self.sample_length = 30  # seconds
        self.eeg_sample_length = self.eeg_frequency * self.sample_length

    def __len__(self):
        return self.subject_count * self.blocks_per_subject * self.samples_per_block
        # first fMRI volume at t=2s

    def __getitem__(self, idx):
        # Computing subject, block, and block-level sample index of sample
        subject = idx // self.samples_per_subject
        block = idx % self.samples_per_subject // self.samples_per_block
        sample_in_block = idx % self.samples_per_block

        if subject >= 3:
            subject += 1

        # Computing sample data location in block recording
        eeg_start_index = sample_in_block * self.eeg_sample_step
        eeg_end_index = eeg_start_index + self.eeg_sample_length
        fmri_index = sample_in_block + self.skipped_samples_per_block

        # Loading EEG and fMRI data from sample block
        subject_path = self.root_dir + '/sub' + f'{(subject + 1):03}/'
        eeg_path = subject_path + 'EEG/' + 'task' + f'{(block // 3 + 1):03}' + \
            '_run' + f'{(block % 3 + 1):03}/' + 'EEG_rereferenced.mat'
        mask_path = subject_path + 'BOLD/mask.nii'

        if block < 3:
            block_type = 'auditory'
        else:
            block_type = 'visual'
	    
        fmri_path = subject_path + 'BOLD/swarsub-' + f'{(subject + 1):02}_task-' + \
            block_type + 'oddballwithbuttonresponsetotargetstimuli_run-' + \
            f'{(block % 3 + 1):02}_' + 'bold.nii'

        eeg_block_data = loadmat(eeg_path)['data_reref']
        fmri_block_data = nib.load(fmri_path).get_fdata()
        mask_data = nib.load(mask_path).get_fdata()

        # Extract relevant sample data from block
        eeg_block_data = eeg_block_data[:34]
        fmri_block_data = np.moveaxis(fmri_block_data, -1, 0)

        eeg_sample_data = np.array(
            [eeg_block_data[channel][eeg_start_index:eeg_end_index] for channel in range(len(eeg_block_data))])
        fmri_sample_data = fmri_block_data[fmri_index]

        # Standardize EEG and fMRI sample data using pre-computed values
        eeg_sample_data = (eeg_sample_data + 2286) / 4200
        fmri_sample_data = (fmri_sample_data + 419) / 4555
        #fmri_sample_data += 500
        #fmri_sample_data = (np.log10(fmri_sample_data) - 1.4) / 3.76
        #fmri_sample_data = fmri_sample_data * 2 - 1

        # Convert to Pytorch tensors
        eeg_sample_data = torch.from_numpy(eeg_sample_data.astype('float32'))
        fmri_sample_data = torch.from_numpy(fmri_sample_data.astype('float32'))
        subject = torch.from_numpy(np.array([subject / 100]).astype('float32'))
        mask_data = torch.from_numpy(mask_data)

        return [[eeg_sample_data, subject], [fmri_sample_data, mask_data]]


def fmri_preview(volume):
    """Preview central slices of an fMRI volume across its 3 dimensions."""
    slice_sagital = volume[26, :, :]
    slice_coronal = volume[:, 31, :]
    slice_horizontal = volume[:, :, 26]

    fig, axes = plt.subplots(1, 3)
    for i, slice in enumerate([slice_sagital, slice_coronal, slice_horizontal]):
        axes[i].imshow(slice.T, cmap="viridis", origin="lower")

    plt.show()


def eeg_preview(frame):
    """Preview electrophysiological data recorded by each channel in an EEG frame."""
    fig, axes = plt.subplots(len(frame), 1)
    for channel in range(len(frame)):
        axes[channel].plot(range(len(frame[channel])), frame[channel])

    plt.show()


def compute_standardization_parameters(dataset):
    """Compute mean and standard deviation of all EEG and fMRI data"""
    full_eeg_data = []
    full_fmri_data = []

    # Load all EEG and fMRI data
    for subject in range(dataset.subject_count):
        for block in range(dataset.blocks_per_subject):
            subject_path = dataset.root_dir + '/sub' + f'{(subject + 1):03}/'
            block_path = 'task' + f'{(block // 3 + 1):03}' + \
                '_run' + f'{(block % 3 + 1):03}/'
            eeg_path = subject_path + 'EEG/' + block_path + 'EEG_rereferenced.mat'
            fmri_path = subject_path + 'BOLD/' + block_path + 'bold_mcf_brain.nii.gz'

            eeg_block_data = loadmat(eeg_path)['data_reref']
            fmri_block_data = nib.load(fmri_path).get_fdata()

            full_eeg_data += [eeg_block_data]
            full_fmri_data += [fmri_block_data]

    print('EEG', np.mean(full_eeg_data), np.std(full_eeg_data))
    print('fMRI', np.mean(full_fmri_data), np.std(full_fmri_data))


def compute_normalization_parameters(dataset):
    """Compute min and max of all EEG and fMRI data"""
    full_eeg_data = []
    full_fmri_data = []

    # Load all EEG and fMRI data
    for subject in range(dataset.subject_count):
        for block in range(dataset.blocks_per_subject):
            # Loading EEG and fMRI data from sample block
            subject_path = dataset.root_dir + '/sub' + f'{(subject + 1):03}/'
            eeg_path = subject_path + 'EEG/' + 'task' + f'{(block // 3 + 1):03}' + \
                '_run' + f'{(block % 3 + 1):03}/' + 'EEG_rereferenced.mat'

            if block < 3:
                block_type = 'auditory'
            else:
                block_type = 'visual'
	    
            fmri_path = subject_path + 'BOLD/warsub-' + f'{(subject + 1):02}_task-' + \
                block_type + 'oddballwithbuttonresponsetotargetstimuli_run-' + \
                f'{(block % 3 + 1):02}_' + 'bold.nii'

            eeg_block_data = loadmat(eeg_path)['data_reref']
            fmri_block_data = nib.load(fmri_path).get_fdata()

            full_eeg_data += [eeg_block_data]
            full_fmri_data += [fmri_block_data]

    print('EEG', np.min(full_eeg_data), np.max(full_eeg_data))
    print('fMRI', np.min(full_fmri_data), np.max(full_fmri_data))


def extract_channel_data(dataset):
    eeg_channel_data = [[] for _ in range(34) ]
    
    # Load all EEG and fMRI data
    for subject in range(dataset.subject_count):
        for block in range(dataset.blocks_per_subject):
            subject_path = dataset.root_dir + '/sub' + f'{(subject + 1):03}/'
            block_path = 'task' + f'{(block // 3 + 1):03}' + \
                '_run' + f'{(block % 3 + 1):03}/'
            eeg_path = subject_path + 'EEG/' + block_path + 'EEG_rereferenced.mat'

            eeg_block_data = loadmat(eeg_path)['data_reref']

            for i in range(34):
                eeg_channel_data[i] += [eeg_block_data[i]]

    return eeg_channel_data
