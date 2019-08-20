"""data loader, ref: tutorial yunjey"""
import os
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image


def get_loader(image_path, image_size=227, batch_size=64, num_workers=2):

    
    # @todo(chuanzi): check imagenet data params
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    dataset = ImageFolder(
        image_path,
        transforms.Compose([
            transforms.RandomSizedCrop(image_size), # @diff: transforms.Scale or crop?
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    return data_loader