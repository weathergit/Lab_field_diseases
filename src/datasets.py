# -*- coding: utf-8 -*-
# @Time    : 2022/10/24 下午8:57
# @Author  : weather
# @Software: PyCharm

from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import os
import json
from pprint import pprint


def data_transforms():
    # standard tranform: resize, ToTensor and Normaliation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


def load_datasets(datasets_folder, data_transforms, batchsize):
    images_datasets = {x: datasets.ImageFolder(os.path.join(datasets_folder, x),
                                               data_transforms[x]) for x in ['train', 'valid']}
    dataloaders = {x: DataLoader(images_datasets[x], batch_size=batchsize, shuffle=True, num_workers=5) for x in
                   ['train', 'valid']}

    dataset_sizes = {x: len(images_datasets[x]) for x in ['train', 'valid']}

    print('Using {} images for training and {} images for validation'.format(dataset_sizes['train'],
                                                                             dataset_sizes['valid']))
    class_names = images_datasets['train'].class_to_idx
    cla_dict = dict((val, key) for key, val in class_names.items())
    json_str = json.dumps(cla_dict, indent=4)
    pprint(class_names)

    out_folder = os.path.join(datasets_folder, 'output')
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    jsonname = os.path.join(out_folder, 'class_indices.json')
    with open(jsonname, 'w') as json_file:
        json_file.write(json_str)
    print('class indices have been written into output folder')
    return dataset_sizes, dataloaders, class_names


if __name__ == "__main__":
    datasets_folder = './field/'
    data_transforms = data_transforms()
    dataset_sizes, dataloaders = load_datasets(datasets_folder, data_transforms)