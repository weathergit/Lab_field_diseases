# -*- coding: utf-8 -*-
# @Time    : 2022/10/24 下午11:14
# @Author  : weather
# @Software: PyCharm
import os
import subprocess


def run_command():
    """
    train lab, field, mixed on total datasets.
    :return:
    """
    train_folders = ['../data/field/', '../data/lab/', '../data/mixed/']
    out_folders = ['../data/field/output/', '../data/lab/output/', '../data/mixed/output/']
    model_names = ["mobile3s", "mobile3l",
                   'res18', 'res34', 'res50', 'res101', 'res152',
                   'effb0', 'effb1', 'effb2', 'effb3', 'effb4', 'effb5', 'effb6', 'effb7',
                   'vgg11', 'vgg13', 'vgg16', 'vgg19',
                   'dense121', 'dense161', 'dense169', 'dense201',
                   'google', 'incep']
    for fld, out in zip(train_folders, out_folders):
        for name in model_names:
            commond = ['python', 'train.py', '-i', fld, '-o', out, '-m', name]
            subprocess.run(commond)


def run_command2(train_folders='./plant_individual/field/'):
    """
    train individual plants.
    :return:
    """
    models_names = ['effi0', 'mobilev3', 'res34']
    for sub in os.listdir(train_folders):
        in_path = os.path.join(train_folders, sub)
        out_path = os.path.join(train_folders, sub, 'output/')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for model in models_names:
            print(in_path)
            command = ['python', 'train.py', '-i', in_path, '-o', out_path, '-m', model]
            subprocess.run(command)


def run_command3():
    """
    train partial models
    :return:
    """
    root = './partial/'
    models_names = ['effi0', 'mobilev3', 'res34']
    for sub in os.listdir(root):
        train_folder = os.path.join(root, sub)
        out_path = os.path.join(train_folder, 'output/')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for model in models_names:
            print(train_folder)
            command = ['python', 'train.py', '-i', train_folder, '-o', out_path, '-m', model]
            subprocess.run(command)


if __name__ == "__main__":
    run_command()