# -*- coding: utf-8 -*-
# @Time    : 2022/10/25 下午12:26
# @Author  : weather
# @Software: PyCharm


import glob
import os
import shutil
from pprint import pprint


def find_all_file(root_dir):
    criterion = os.path.join(root_dir, '*', '*')
    src_folder = glob.glob(criterion)
    return src_folder


def copy_folder(folder_list, outdir, outfoler):
    for fld in folder_list:
        target = fld.split('/')[2]
        crop = fld.split('/')[3].split('_')[0]
        diseases = fld.split('/')[-1]
        if target == 'output':
            pass
        else:
            outpath = os.path.join(outdir, outfoler, crop, target, diseases)
            print(fld, outpath)
            shutil.copytree(fld, outpath)


if __name__ == "__main__":
    # folderlist = find_all_file('./field/')
    # copy_folder(folderlist, './plant_individual/', 'field')
    # folderlist = find_all_file('./lab/')
    # copy_folder(folderlist, './plant_individual/', 'lab')

    folderlist = find_all_file('./mixed/')
    copy_folder(folderlist, './plant_individual/', 'mixed')
