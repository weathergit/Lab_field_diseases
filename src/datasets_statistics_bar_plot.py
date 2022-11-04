# -*- coding: utf-8 -*-
# @Time    : 2022/10/24 下午3:49
# @Author  : weather
# @Software: PyCharm

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def bar_plot(field_folder, lab_folder):
    def get_data(root_dir):
        image_number = {}
        for fn in os.listdir(root_dir):
            search = os.path.join(root_dir, fn, '*')
            images = glob.glob(search)
            image_number[fn] = len(images)
        df = pd.DataFrame.from_dict({'diseases': image_number.keys(),
                                     'image_number': image_number.values()})
        return df

    def field_lab_df(field_floder, lab_folder):
        field_df = get_data(field_floder)
        field_df.columns = ['diseases', 'Field']
        lab_df = get_data(lab_folder)
        lab_df.columns = ['diseases', 'Lab']
        res = pd.merge(field_df, lab_df)
        return res

    res = field_lab_df(field_folder, lab_folder)
    ax = res.plot(kind='bar', stacked=True,
                  color=['Tab:red', 'Tab:blue'])
    xlabel = res.diseases.str.replace('_', ' ')
    ax.set_xticklabels(xlabel)
    ax.bar_label(ax.containers[0],
                 label_type='center', fontsize=6)
    ax.bar_label(ax.containers[1],
                 label_type='center', fontsize=6)
    plt.legend(fontsize=10, frameon=True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    field_folder = '../data/original_datasets/merged_datasets/field/'
    lab_folder = '../data/original_datasets/merged_datasets/lab/'
    bar_plot(field_folder, lab_folder)
