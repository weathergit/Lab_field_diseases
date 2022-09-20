import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['ytick.labelsize'] = 12

result = pd.read_csv('../data/total_cross_test_frame.csv')

field = result[result['sources'] == 'field']

lab = result[result['sources'] == 'lab']

mixed = result[result['sources'] == 'mixed']

colors = {'lab': '#AE3D3A', 'mixed': '#3382A3', 'field': '#304E6C'}

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(5, 9), facecolor='w')
# lab on min field
axs[0].bar(x=np.arange(3)-0.1, height=lab['acc'].values[0::2], width=0.2, color=colors['field'], label='Field')
axs[0].bar(x=np.arange(3)+0.1 , height=lab['acc'].values[1::2], width=0.2, color=colors['mixed'], label='Mixed')
axs[0].set_ylim(0, 1)
axs[0].set_ylabel('Lab', fontsize=16)
axs[0].legend(loc=9, ncol=2, frameon=False, fontsize=12)

# Mixed on lab field
axs[1].bar(x=np.arange(3)-0.1, height=mixed['acc'].values[0::2], width=0.2, color=colors['field'], label='Field')
axs[1].bar(x=np.arange(3) + 0.1, height=mixed['acc'].values[1::2], width=0.2, color=colors['lab'], label='Lab')
axs[1].set_ylim(0, 1)
axs[1].set_ylabel('Mixed', fontsize=16)
axs[1].legend(loc=9, ncol=2, frameon=False, fontsize=12)
# Field on mixed lab
axs[2].bar(x=np.arange(3)-0.1, height=field['acc'].values[0::2], width=0.2, color=colors['lab'], label='Lab')
axs[2].bar(x=np.arange(3) + 0.1, height=field['acc'].values[1::2], width=0.2, color=colors['mixed'], label='Mixed')
axs[2].set_ylim(0, 1)
axs[2].legend(loc=9, ncol=2, frameon=False, fontsize=12)
axs[2].set_ylabel('Field', fontsize=16)
models_labels = ['EfficientNet B0', 'MobileNet V3s', 'ResNet34']
plt.xticks(np.arange(3), labels=models_labels, fontsize=12)
plt.yticks(fontsize=12)
# plt.xlim(-0.2, 3.4)
plt.tight_layout()
plt.savefig('../fig/cross_test.png', dpi=300)
plt.show()
