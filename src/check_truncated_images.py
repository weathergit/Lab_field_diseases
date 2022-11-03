# -*- coding: utf-8 -*-
# @Time    : 2022/10/24 下午10:43
# @Author  : weather
# @Software: PyCharm
import os.path

from PIL import Image
import glob


def find_truncted(folder):
    ctiter = os.path.join(folder, '*', '*', '*')
    images = glob.glob(ctiter)
    trunc_list = []
    for path in images:
        try:
            img = Image.open(path)
            img = img.convert('RGB')
        except:
            print(path)
            trunc_list.append(path)
    return trunc_list


def mvfile(imglist):
    for img in imglist:
        os.remove(img)


if __name__ == "__main__":
    field_folder = './field/'
    lab_folder = './lab/'
    mixed_folder = './mixed/'
    imglist = find_truncted(field_folder)
    imglist = find_truncted(mixed_folder)
    imglist = find_truncted(lab_folder)
    print(imglist)
    # mvfile(imglist)
    # find_truncted(lab_folder)
    ## noting that the truncted file also exits in mixed folders
    # os.remove('./field/valid/Potato_early_blight/1_2 (24).jpg')
    # os.remove('.mixed/valid/Potato_early_blight/1_2 (24).jpg')
