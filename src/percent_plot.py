#!/usr/bin/env python
# encoding: utf-8
"""
# @Time    : 2022/7/4 12:04
# @Author  : weather
# @Software: PyCharm
"""

"""
this script  plot the scatter/lines with different x and legend
x: [10%, 20%,......,100%]
legends: Efficient Net , MobileNet, ResNet
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
    df = pd.read_csv('../data/part_val_test_acc.csv')
    p1 = sns.lineplot(x='percent', y='val acc', hue='models', data=df, marker='*', lw=2, ms=12, legend=True)
    sns.lineplot(x='percent', y='test acc', hue='models', data=df, marker='o', lw=1.5, ms=6, alpha=0.7, legend=False)
    handles, _ = p1.get_legend_handles_labels()
    l1 = plt.legend(handles=handles, labels=['EfficientNet B0', 'MobileNet V3s', 'ResNet 34'], frameon=False,
                    fontsize=12, bbox_to_anchor=[0.55, 0.5])
    custom_scatter = [plt.scatter(0, 0, marker='*', label='Validation Accuracy', color='k'),
                      plt.scatter(0, 0, marker='o', label='Test Accuracy', color='k')]
    l2 = plt.legend(handles=custom_scatter, fontsize=12, frameon=False,bbox_to_anchor=[0.55, 0.3])

    p1.add_artist(l1)
    p1.add_artist(l2)

    plt.xticks(ticks=np.arange(1, 11, 1), labels=['{}%'.format(i) for i in range(10, 110, 10)], fontsize=12)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.savefig('../fig/lab_add2field_val_test_acc.png', dpi=300)
    plt.savefig('../fig/lab_add2field_val_test_acc.tiff', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
