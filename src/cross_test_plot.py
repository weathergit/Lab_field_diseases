import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import bar

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['ytick.labelsize'] = 12


def main(lab, mixed, field):
    colors = {'lab': '#AE3D3A', 'mixed': '#3382A3', 'field': '#304E6C'}
    models_labels = ['EfficientNet B0', 'MobileNet V3s', 'ResNet34']

    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(15, 5), facecolor='w')
    # lab on min field
    axs[0].bar(x=np.arange(3) - 0.1, height=lab['acc'].values[0::2], width=0.2, color=colors['field'],
               label='Test on Field')
    axs[0].bar(x=np.arange(3) + 0.1, height=lab['acc'].values[1::2], width=0.2, color=colors['mixed'],
               label='Test on Mixed')
    axs[0].set_ylim(0, 1)
    axs[0].set_ylabel('Train on Lab', fontsize=20)
    # axs[0].legend(loc=9, ncol=2, frameon=False, fontsize=14)
    axs[0].set_xticks(np.arange(3), labels=models_labels, fontsize=15)

    # Mixed on lab field
    axs[1].bar(x=np.arange(3) - 0.1, height=mixed['acc'].values[0::2], width=0.2, color=colors['field'],
               label='Test on Field')
    axs[1].bar(x=np.arange(3) + 0.1, height=mixed['acc'].values[1::2], width=0.2, color=colors['lab'],
               label='Test on Lab')
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel('Train on Mixed', fontsize=20)
    # axs[1].legend(loc=9, ncol=2, frameon=False, fontsize=14)
    axs[1].set_xticks(np.arange(3), labels=models_labels, fontsize=15)
    # Field on mixed lab
    axs[2].bar(x=np.arange(3) - 0.1, height=field['acc'].values[0::2], width=0.2, color=colors['lab'], label='Test on '
                                                                                                             'Lab')
    axs[2].bar(x=np.arange(3) + 0.1, height=field['acc'].values[1::2], width=0.2, color=colors['mixed'], label='Test '
                                                                                                               'on '
                                                                                                               'Mixed')
    axs[2].set_ylim(0, 1)
    # axs[2].legend(loc=9, ncol=2, frameon=False, fontsize=14)
    axs[2].set_ylabel('Train on Field', fontsize=20)

    axs[2].set_xticks(np.arange(3), labels=models_labels, fontsize=15)

    legends = [bar([0], [0], color=colors['field'], label='Test on Field'),
               bar([0], [0], color=colors['mixed'], label='Test on Mixed'),
               bar([0], [0], color=colors['lab'], label='Test on Lab')]
    axs[2].legend(handles=legends, loc=0, frameon=False, fontsize=16)

    plt.yticks(fontsize=14)
    # plt.xlim(-0.2, 3.4)
    plt.tight_layout()
    plt.savefig('../fig/cross_test_row.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    result = pd.read_csv('../data/total_cross_test_frame.csv')

    field_df = result[result['sources'] == 'field']

    lab_df = result[result['sources'] == 'lab']

    mixed_df = result[result['sources'] == 'mixed']

    main(lab_df, mixed_df, field_df)
