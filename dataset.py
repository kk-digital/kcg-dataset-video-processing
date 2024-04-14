import torch
from PIL import Image
import numpy as np
from torchvision.transforms.v2 import functional as VF
from torchvision.transforms.v2 import RandomResizedCrop

class Dataset(torch.utils.data.Dataset):

    def __init__(self, file_paths, target_size):
        
        self.file_paths = file_paths
        self.target_size = target_size
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        
        file_path = self.file_paths[index]

        image = Image.open(file_path).convert('RGB')

        scale = min((self.target_size / min(image.size)) ** 2, .5)

        params = RandomResizedCrop.get_params(image, scale=(scale, 1), ratio=(1.,1.))
        image = VF.resized_crop(image, *params, size=self.target_size, interpolation=VF.InterpolationMode.BICUBIC, antialias=True)

        image = np.array(image)
        
        return image
    