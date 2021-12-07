from torch.utils.data.dataset import Dataset
import torch
from torchvision import transforms

import numpy as np
from pathlib import Path
import pandas as pd

class DCASE(Dataset):
    def __init__(self, root_dir: str, clip_duration: int):
        self._root_dir = Path(root_dir)
        self._labels = pd.read_csv((self._root_dir / 'labels.csv'), names=['file', 'label'])
        # print(self._labels.label.astype('category'))
        self._labels['label'] = self._labels.label.astype('category').cat.codes.astype('int') #create categorical labels
        self._clip_duration = clip_duration
        self._total_duration = 30 #DCASE audio length is 30s

        self._data_len = len(self._labels)

    def __getitem__(self, index):
        #reading spectrograms
        filename, label = self._labels.iloc[index]
        filepath = self._root_dir / 'audio'/ filename
        spec = torch.from_numpy(np.load(filepath))

        #splitting spec
        spec = self.__trim__(spec)
        return spec, label

    def __trim__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Trims spectrogram into multiple clips of length specified in self._num_clips
        :param spec: tensor containing spectrogram of full audio signal of shape [1, 60, 1501]
        :return: tensor containing stacked spectrograms of shape [num_clips, 60, clip_length] ([10, 60, 150] with 3s clips)
        """
        time_steps = spec.size(-1)
        self._num_clips = self._total_duration // self._clip_duration
        time_interval = int(time_steps // self._num_clips)
        all_clips = []
        for clip_idx in range(self._num_clips):
            start = clip_idx * time_interval
            end = start + time_interval
            spec_clip = spec[:, start:end]
            #spec_clip = torch.squeeze(spec_clip)
            all_clips.append(spec_clip)

        specs = torch.stack(all_clips)
        return specs

    def get_num_clips(self) -> int:
        """
        Gets number of clips the raw audio has been split into
        :return: self._num_clips of type int
        """
        return self._num_clips

    def __len__(self):
        return self._data_len

class DCASE_clip(DCASE):
    
    def __init__(self, root_dir: str, clip_duration: int, normData = False, priorNorm = None):
        super().__init__(root_dir, clip_duration)
        self._num_clips = self._total_duration // self._clip_duration
        self.normData = normData
        if self.normData:
            if priorNorm is not None:
                print("Using proir norm.")
                self.specs_mean, self.specs_std = priorNorm
            else:    
                self.norm_data()
    
    def __getitem__(self, clip_index):
        spec_index, clip_offset = divmod(clip_index, self._num_clips)
        spec, label = self.get_spec_index(spec_index)
        if self.normData:
            spec = (spec - self.specs_mean)/self.specs_std
        #splitting spec
        spec = super().__trim__(spec)
        return np.expand_dims(spec[clip_offset], axis=0), label
        
    def __len__(self):
        return self._data_len * self._num_clips
    
    def get_spec_index(self, spec_index):
        filename, label = self._labels.iloc[spec_index]
        filepath = self._root_dir / 'audio' / filename
        spec = torch.from_numpy(np.load(filepath))
        return spec, label
    
    def norm_data(self):
        print(self._data_len)
        spec_shape = self.get_spec_index(0)[0].shape
        print("Computing norm.")
        specs = np.zeros((self._data_len, spec_shape[0], spec_shape[1]), dtype='f')
        for i in range(0 ,self._data_len):
            specs[i], _ =  self.get_spec_index(i)
        self.specs_mean = specs.mean(axis=0)
        self.specs_std = specs.std(axis=0)
        print("Done.")
        return specs
        
    def prior_norm(self):
        return self.specs_mean, self.specs_std
        
        