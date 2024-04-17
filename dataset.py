import torch
from PIL import Image, ImageOps
import numpy as np
from torchvision.transforms.v2 import functional as VF
from torchvision.transforms.v2 import RandomResizedCrop

def pad_to_square(pil_img: Image.Image, fill_color=0):
    width, height = pil_img.size

    if width == height:
        return pil_img
    elif width > height:
        padding = (0, (width - height) // 2)
    else:
        padding = ((height - width) // 2, 0)

    result = ImageOps.expand(pil_img, border=padding, fill=fill_color)
    return result

def crop_to_square(pil_img: Image.Image):
    width, height = pil_img.size
    
    if width == height:
        return pil_img
    elif width > height:
        left = (width - height) // 2
        right = left + height
        top = 0
        bottom = height
    else:
        top = (height - width) // 2
        bottom = top + width
        left = 0
        right = width

    result = pil_img.crop((left, top, right, bottom))
    return result

class Dataset(torch.utils.data.Dataset):

    def __init__(self, file_paths: list, target_size: int):
        
        self.file_paths = file_paths
        self.target_size = target_size
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index: int):
        
        file_path = self.file_paths[index]

        image = Image.open(file_path).convert('RGB')

        scale = min((self.target_size / min(image.size)) ** 2, .5)

        params = RandomResizedCrop.get_params(image, scale=(scale, 1), ratio=(1.,1.))
        image = VF.resized_crop(image, *params, size=self.target_size, interpolation=VF.InterpolationMode.BICUBIC, antialias=True)

        image = np.array(image)
        
        return image
    