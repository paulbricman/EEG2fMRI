from torch.utils.data import Dataset

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
        self.samples_per_trial = 155
        self.trials_per_subject = 6
        self.samples_per_subject = self.samples_per_trial * self.trials_per_subject
        self.subject_count = len(os.listdir())
        self.skipped_samples_per_trial = 170 - self.samples_per_trial
        self.eeg_frequency = 1000  # Hz
        self.fmri_period = 2  # seconds
        self.skipped_eeg_per_trial = self.skipped_samples_per_trial * \
            self.fmri_period * self.eeg_frequency
        self.eeg_sample_step = self.fmri_period * self.eeg_frequency
        self.sample_length = 30  # seconds
        self.eeg_sample_length = self.eeg_frequency * self.sample_length

    def __len__(self):
        return self.subject_count * self.trials_per_subject * self.samples_per_trial

    def __getitem__(self, idx):
        # Computing subject, trial, and trial-level sample index of sample
        subject = idx // self.samples_per_subject
        trial = idx % self.samples_per_subject // self.samples_per_trial
        sample_in_trial = idx % self.samples_per_trial

        # Computing sample data location in trial recording
        eeg_start_index = sample_in_trial * self.eeg_sample_step
        eeg_end_index = eeg_start_index + self.eeg_sample_length
        fmri_index = sample_in_trial + self.skipped_samples_per_trial

        # Loading EEG and fMRI data from sample trial
        subject_path = self.root_dir + '/ds116_sub' + \
            f'{(subject + 1):03}' + '/sub' + f'{(subject + 1):03}/'
        trial_path = 'task' + f'{(trial // 3 + 1):03}' + \
            '_run' + f'{(trial % 3 + 1):03}/'
        eeg_path = subject_path + 'EEG/' + trial_path + 'EEG_rereferenced.mat'
        fmri_path = subject_path + 'BOLD/' + trial_path + 'bold_mcf_brain.nii.gz'

        eeg_trial_data = loadmat(eeg_path)['data_reref']
        fmri_trial_data = nib.load(fmri_path).get_fdata()

        # Extract relevant sample data from trial
        eeg_trial_data = eeg_trial_data[:34]  # 33 if no BCG
        fmri_trial_data = np.moveaxis(fmri_trial_data, -1, 0)

        eeg_sample_data = np.array(
            [eeg_trial_data[channel][eeg_start_index:eeg_end_index] for channel in range(len(eeg_trial_data))])
        fmri_sample_data = fmri_trial_data[fmri_index]

        return [eeg_sample_data, fmri_sample_data]


def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

    plt.show()


def fmri_preview(volume):
    slice_sagital = volume[32, :, :]
    slice_coronal = volume[:, 32, :]
    slice_horizontal = volume[:, :, 16]

    show_slices([slice_sagital, slice_coronal, slice_horizontal])


def eeg_preview(frame):
    fig, axes = plt.subplots(len(frame), 1)
    for channel in range(len(frame)):
        axes[channel].plot(range(len(frame[channel])), frame[channel])

    plt.show()


dataset = OddballDataset('../../OddballData')
sample = dataset[70]
eeg_preview(sample[0])
fmri_preview(sample[1])
