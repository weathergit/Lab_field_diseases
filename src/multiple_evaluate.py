# -*- coding: utf-8 -*-
# @Time    : 2022/10/29 下午12:08
# @Author  : weather
# @Software: PyCharm

import os
import subprocess


def evaluate1(root_dir='./plant_individual/field/'):
    """
    evaluate individual
    :return:
    """
    for sub in os.listdir(root_dir):
        inpath = os.path.join(root_dir, sub)
        command = ['python', 'evaluate.py', '--input', inpath, '--outname', 'predict_test']
        subprocess.run(command)


def evaluate2():
    """
    partial
    :return:
    """
    root = './partial/'
    for part in os.listdir(root):
        inpath = os.path.join(root, part)
        command = ['python', 'evaluate.py', '--input', inpath, '--outname', 'predict_test']
        subprocess.run(command)


def cross_tested_run():
    data_folders = ['./lab/', './mixed/', './field/']
    for dfld in data_folders:
        list_copy = data_folders.copy()
        list_copy.remove(dfld)
        for ifld in list_copy:
            outname = ifld[2:-1] + '_' + dfld[2:-1]
            command = ['python', 'evaluate.py', '--input', ifld, '--data_folder', dfld, '--output', './cross_tested/',
                       '--outname', outname]
            subprocess.run(command)


def partial_model_on_field():
    root = './partial/'
    for sub in os.listdir(root):
        inpath = os.path.join(root, sub)
        data_folder = './field/'
        outpath = os.path.join('./part_test_on_field/', sub + '/')
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        command = ['python', 'evaluate.py', '--input', inpath, '--data_folder', data_folder, '--output',
                   outpath, '--outname', 'partial_test']
        subprocess.run(command)


if __name__ == "__main__":
    # partial_model_on_field()
    # evaluate1(root_dir='./plant_individual/lab/')
    # evaluate1(root_dir='./plant_individual/mixed/')
    evaluate1()