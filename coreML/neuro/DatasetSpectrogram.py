import numpy as np
import pandas as pd
import utils
from typing import Tuple, Callable

import torch
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset


__all__ = ['DatasetSpectrogram']


class DatasetSpectrogram(Dataset):

    def __init__(self, 
                df: pd.DataFrame, 
                path_audio: str, 
                transforms: Compose = None, 
                audio2: Callable[..., np.ndarray] = None):
        self.files = [(row['filename'], row['disease']) for col, row in df.iterrows()]
        self.length = len(self.files)
        self.path_audio = path_audio
        if transforms == None:
            self.transforms = Compose([
                ToTensor()
            ])
        else:
            self.transforms = transforms
        if audio2 == None:
            self.audio = utils.audio2spec
        else:
            self.audio = audio2

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename, label = self.files[index]
        filename, self.path_audio + '/' + filename
        audio_tensor = utils.normalize(self.audio2(filename))
        return (self.transforms(audio_tensor), torch.tensor([label]))
    
    def __len__(self):
        return self.length