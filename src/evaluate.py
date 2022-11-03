# -*- coding: utf-8 -*-
# @Time    : 2022/10/25 下午12:26
# @Author  : weather
# @Software: PyCharm

import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torchvision import models
from glob import glob
from collections import defaultdict
import pandas as pd
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_transform():
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return data_transform


def get_img2tensor(rootdir, data_transform):
    img_path_list = []
    for plant in os.listdir(rootdir):
        path = rootdir + plant + "/*"
        images = glob(path)
        img_path_list.extend(images)

    img_list = []
    for img_path in img_path_list:
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = data_transform(img)
        img_list.append(img)

    return img_path_list, img_list


def set_indices(json_path):
    # read class_indict
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indics = json.load(json_file)
    return class_indics


def set_models(num_cla, model_name, weights_path):
    # create model
    if model_name == 'res34':
        model = models.resnet34(num_classes=num_cla).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    if model_name == 'mobilev3':
        model = models.mobilenet_v3_small(num_classes=num_cla).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    if model_name == 'effi0':
        model = models.efficientnet_b0(num_classes=num_cla).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


def predict(model, img_path_list, img_list, class_indices, outdir, outname):
    model.eval()
    model_name = model._get_name()
    result = defaultdict(list)
    for i in range(0, len(img_list), 100):
        batch_img = torch.stack(img_list[i:i + 100], dim=0)
        with torch.no_grad():
            output = model(batch_img.to(device))
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                result['img'].append(img_path_list[i + idx])
                result['pro'].append(pro.cpu().numpy())
                index = cla.cpu().numpy()
                result['cla'].append(class_indices[str(index)])
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[i + idx],
                                                                 class_indices[str(index)],
                                                                 pro.cpu().numpy()))
    df = pd.DataFrame.from_dict(result)
    outname = outdir + model_name + '_' + outname + ".csv"
    df.to_csv(outname)


def main(args):
    indir = os.path.join(args.input, 'test/')
    indices = set_indices(os.path.join(args.input, 'output', 'class_indices.json'))
    if args.output is None:
        outdir = os.path.join(args.input, 'output/')
    else:
        outdir = args.output
    data_trans = set_transform()
    path_list, img_list = get_img2tensor(indir, data_trans)
    num_cls = len(indices.keys())
    model_names = ['res34', 'mobilev3', 'effi0']
    weights = [os.path.join(args.input, 'output', 'ResNet.pth'), os.path.join(args.input, 'output', 'MobileNetV3.pth'),
               os.path.join(args.input, 'output', 'EfficientNet.pth')]
    for name, weight in zip(model_names, weights):
        model = set_models(num_cls, name, weight)
        predict(model, path_list, img_list, indices, outdir, args.outname)


def cross_tested(args):
    if not args.data_folder:
        indir = os.path.join(args.input, 'test/')
    else:
        indir = os.path.join(args.data_folder, 'test/')
    indices = set_indices(os.path.join(args.input, 'output', 'class_indices.json'))
    if args.output is None:
        outdir = os.path.join(args.input, 'output/')
    else:
        outdir = args.output
    data_trans = set_transform()
    path_list, img_list = get_img2tensor(indir, data_trans)
    num_cls = len(indices.keys())
    model_names = ['res34', 'mobilev3', 'effi0']
    weights = [os.path.join(args.input, 'output', 'ResNet.pth'), os.path.join(args.input, 'output', 'MobileNetV3.pth'),
               os.path.join(args.input, 'output', 'EfficientNet.pth')]
    for name, weight in zip(model_names, weights):
        model = set_models(num_cls, name, weight)
        predict(model, path_list, img_list, indices, outdir, args.outname)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input images folder')
    parser.add_argument('--output', type=str, help='output dir')
    parser.add_argument('--outname', type=str, help='output csv name')
    parser.add_argument('--data_folder', type=str, default=None, help='output csv name')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()
    cross_tested(args)
