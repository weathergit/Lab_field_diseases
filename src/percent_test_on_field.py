#!/usr/bin/env python
# encoding: utf-8
"""
# @Time    : 2022/7/4 12:04
# @Author  : weather
# @Software: PyCharm
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['ytick.labelsize'] = 12

colors = ['#F65429', '#EED364', '#3382A3']
sns.set_palette(sns.color_palette(colors))


def main():
    df = pd.read_csv('../data/field_test_acc.csv')
    sns.lineplot(x=df['percent'], y=df['acc'], hue='models', linewidth=2,
                 marker='*', ms=12, data=df)
    # sns.lineplot(x=df['percent'], y=df['test acc'], hue='models', linewidth=1.5,
    #              marker='o', ms=6, legend=None, alpha=0.7, data=df)
    # plt.ylim(0.75, 0.9)
    plt.legend(labels=['EfficientNet B0', 'MobileNet V3s', 'ResNet 34'], frameon=False, fontsize=12)
    plt.xticks(ticks=np.arange(1, 11, 1), labels=['{}%'.format(i) for i in range(10, 110, 10)], fontsize=12)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.savefig('../fig/percent_test_on_field.png', dpi=300)
    plt.savefig('../fig/percent_test_on_field.tiff', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()