from torch.utils.data import Dataset

import nibabel as nib
from scipy.io import loadmat

import os

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

        self.samples_per_trial = 155
        self.trials_per_subject = 6
        self.samples_per_subject = self.samples_per_trial * self.trials_per_subject
        self.subject_count = len(os.listdir())
        self.skipped_samples_per_trial = 170 - self.samples_per_trial
        self.eeg_frequency = 1000
        self.fmri_period = 2
        self.skipped_eeg_per_trial = self.skipped_samples_per_trial * self.fmri_period * self.eeg_frequency
        self.eeg_sample_step = self.fmri_period * self.eeg_frequency
        self.sample_length = 30
        self.eeg_sample_length = self.eeg_frequency * self.sample_length

    def __len__(self):
        return self.subject_count * self.trials_per_subject * self.samples_per_trial

    def __getitem__(self, idx):
        subject = idx // self.samples_per_subject
        trial = idx % self.samples_per_subject // self.samples_per_trial
        sample_in_trial = idx % self.samples_per_trial

        eeg_start_index = sample_in_trial * self.eeg_sample_step
        eeg_end_index = eeg_start_index + self.eeg_sample_length
        fmri_index = sample_in_trial + self.skipped_samples_per_trial

        subject_path = self.root_dir + '/ds116_sub' + f'{(subject + 1):03}' + '/sub' + f'{(subject + 1):03}/'
        trial_path = 'task' + f'{(trial // 3 + 1):03}' + '_run' + f'{(trial % 3 + 1):03}/'
        eeg_path = subject_path + 'EEG/' + trial_path + 'EEG_rereferenced.mat'
        fmri_path = subject_path + 'BOLD/' + trial_path + 'bold_mcf_brain.nii.gz'

        eeg_trial_data = loadmat(eeg_path)['data_reref']
        fmri_trial_data = nib.load(fmri_path).get_fdata()

        return [eeg_trial_data.shape, fmri_trial_data.shape]

dataset = OddballDataset('../../OddballData')
print(dataset[600])