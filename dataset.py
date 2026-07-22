import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

def read_16uint_tiff(img_path:Path|str, scale_with_percentile:None|float=None):
    img = Image.open(img_path)
    
    img_np = np.array(img, dtype=np.float32)/65535
    
    if scale_with_percentile!=None:
        img_np /= np.percentile(img_np, scale_with_percentile)

    img_np = np.clip(img_np, 0, 1)
    img_tensor = torch.from_numpy(img_np)
    
    return img_tensor


class StiffnessDataset(Dataset):
    def __init__(self,
                 annotations_file:str,
                 root_dir:Path|str,
                 ch_names=('Nucleus', 'Collagen'),
                 ch_dir_suffix='',
                 ch_name_prefix=('C1-','C2-'),
                 transform=None, target_transform=None):
        '''
        - annotations_file : *.xlsx file with 'Stiffness' and 'Image' (image names) columns,
                            all channel prefixes are removed from 'Image' column before generating
                            image file names for the dataset
        - root_dir : root directory that contains the `annotations_file` and image directories
        - ch_names: tuple containing channel names, `ch_names` must macth names of the image
                   directories in the `root_dir`.
        - ch_dir_suffix: a string used as a suffix for all names in the `ch_names`, e.g.,
                        use ch_dir_suffix="Train_" for folders are named "Train_Nucleus/" and
                        "Train_Collagen"
        - ch_name_prefix: tuple of strings used as prefixes for image file names
        - transform: input/image transfroms
        - target_transform: transforms for the labels (y)
        '''
        def remove_ch_prefix(s):
            for prefix in ch_name_prefix:
                if s.startswith(prefix):
                    s = s.removeprefix(prefix)
                    break
            return s
        
        root_dir = Path(root_dir)
        assert (root_dir/annotations_file).exists(), f"Could not find the {(root_dir/annotations_file)}"
        
        labels_df = pd.read_excel(root_dir/annotations_file)
        labels_df['Image'] = labels_df['Image'].apply(remove_ch_prefix)
        
        for ch, prefix in zip(ch_names,ch_name_prefix):
            labels_df[ch] = labels_df['Image'].apply(lambda x: prefix + x +'.tif')
            
        cols = ['Stiffness']

        cols.extend(ch_names)
        self.img_channels = ch_names
        self.img_labels = labels_df[cols]
        
        self.img_dirs = tuple([root_dir/(ch_k + ch_dir_suffix) for ch_k in ch_names])
        self.transform = transform
        self.target_transform = target_transform

    def __repr__(self):
        desc = f'{self.__class__.__name__}: {self.img_dirs[0].parent.name}'
        desc += f'; {len(self)} labels'
        desc += f'\nchannels: ({", ".join(ch.name for ch in self.img_dirs)})'
        desc += f'; {self.transform}' if self.transform else ''
        desc += f'; {self.target_transform}' if self.target_transform else ''
        return desc

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        stiffness = self.img_labels['Stiffness'].iloc[idx]
        
        img_names = [
            img_dir/k 
            for img_dir,k in zip(self.img_dirs, self.img_labels[self.img_channels].iloc[idx])
        ]
        
        images = [
            read_16uint_tiff(img_name, scale_with_percentile=99)[None, :]
            for img_name  in img_names
        ]
        
        images.append(torch.zeros_like(images[0]))
        
        image = torch.cat(images) 
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            stiffness = self.target_transform(stiffness)
        return image, torch.tensor([stiffness], dtype=torch.float32)


class ValidationDataset(StiffnessDataset):
    def __getitem__(self, idx):
        sample_ch1_name = str(self.img_labels[self.img_channels].iloc[idx, 0])
        stiffness = self.img_labels['Stiffness'].iloc[idx]
        
        img_names = [
            img_dir/k 
            for img_dir,k in zip(self.img_dirs, self.img_labels[self.img_channels].iloc[idx])
        ]
        
        images = [
            read_16uint_tiff(img_name, scale_with_percentile=99)[None, :]
            for img_name  in img_names
        ]
        
        images.append(torch.zeros_like(images[0]))
        
        image = torch.cat(images) 
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            stiffness = self.target_transform(stiffness)

        return image, torch.tensor([stiffness], dtype=torch.float32), sample_ch1_name


class StiffnessDatasetAge(StiffnessDataset):
    def __getitem__(self, idx):
        #sample_ch1_name = str(self.img_labels[self.img_channels].iloc[idx, 0])
        stiffness = self.img_labels['Stiffness'].iloc[idx]
        
        img_names = [
            img_dir/k 
            for img_dir,k in zip(self.img_dirs, self.img_labels[self.img_channels].iloc[idx])
        ]
        
        images = [
            read_16uint_tiff(img_name, scale_with_percentile=99)[None, :]
            for img_name  in img_names
        ]

        if (img_names[0].name)[3] == 'Y':
            images.append(torch.zeros_like(images[0]))
        else:
            images.append(torch.zeros_like(images[0]) + 1.0)
        
        image = torch.cat(images) 
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            stiffness = self.target_transform(stiffness)

        return image, torch.tensor([stiffness], dtype=torch.float32)


class ValidationDatasetAge(StiffnessDataset):
    def __getitem__(self, idx):
        sample_ch1_name = str(self.img_labels[self.img_channels].iloc[idx, 0])
        stiffness = self.img_labels['Stiffness'].iloc[idx]
        
        img_names = [
            img_dir/k 
            for img_dir,k in zip(self.img_dirs, self.img_labels[self.img_channels].iloc[idx])
        ]
        
        images = [
            read_16uint_tiff(img_name, scale_with_percentile=99)[None, :]
            for img_name  in img_names
        ]
        
        if (img_names[0].name)[3] == 'Y':
            images.append(torch.zeros_like(images[0]))
        else:
            images.append(torch.zeros_like(images[0]) + 1.0)
        
        image = torch.cat(images) 
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            stiffness = self.target_transform(stiffness)

        return image, torch.tensor([stiffness], dtype=torch.float32), sample_ch1_name
