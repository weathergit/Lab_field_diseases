# -*- coding: utf-8 -*-
# @Time    : 2022/10/25 下午12:27
# @Author  : weather
# @Software: PyCharm

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os
import numpy as np
from sklearn.metrics import accuracy_score
from pprint import pprint
from sklearn.metrics import confusion_matrix
plt.rcParams['font.family']= 'Times New Roman'


def data_statistics_bar_plot():
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

    field_folder = '../data/original_datasets/merged_datasets/field/'
    lab_folder = '../data/original_datasets/merged_datasets/lab/'
    res = field_lab_df(field_folder, lab_folder)
    res = res.sort_values(by='diseases')
    ax = res.plot(kind='bar', stacked=True,
                  color=['Tab:red', 'Tab:blue'])
    xlabel = res.diseases.str.replace('_', ' ')
    ax.set_xticklabels(xlabel)
    ax.bar_label(ax.containers[0],
                 label_type='center', fontsize=8)
    ax.bar_label(ax.containers[1],
                 label_type='center', fontsize=8)
    plt.legend(fontsize=10, frameon=True)
    plt.xlabel('Diseases name', fontsize=12)
    plt.ylabel('Image number', fontsize=12)
    plt.tight_layout()
    plt.savefig('../fig/original_data_statistics.bar.png', dpi=300)
    plt.show()


def train_acc_loss_plot():
    folders = ['../data/lab/', '../data/field/', '../data/mixed/']
    csv_files = []
    for fld in folders:
        cf = sorted(glob.glob(fld + 'output/*loss*csv'))
        csv_files.extend(cf)

    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    fig, axs = plt.subplots(nrows=3, ncols=3, facecolor='w', figsize=(12, 12))
    axs = axs.flatten()
    for i, path in enumerate(csv_files):
        df = pd.read_csv(path, index_col=0)
        if i < 3:
            df = df.iloc[:15, ]
        xlim = df.shape[0]
        l1 = axs[i].plot(range(1, xlim + 1), df['train_acc'], lw=2, color='#F65429', label='Train Acc')
        l2 = axs[i].plot(range(1, xlim + 1), df['val_acc'], lw=2, color='#DB3725', label='Val Acc')
        axs[i].set_ylim(0, 1.1)
        axs[i].set_yticks(ticks=np.arange(0, 1.05, 0.1))

        ax1 = axs[i].twinx()
        l3 = ax1.plot(range(1, xlim + 1), df['train_loss'], lw=2, color='#3D93C5', label='Train Loss')
        l4 = ax1.plot(range(1, xlim + 1), df['val_loss'], lw=2, color='#3382A3', label='Val Loss')

        lns = l1 + l2 + l3 + l4
        labels = [l.get_label() for l in lns]

        plt.xticks(ticks=range(1, xlim + 1, 3))

        axs[i].grid(ls='-.')

    axs[0].legend(lns, labels, fontsize=12, loc=7)

    axs[0].set_ylabel('Lab', fontsize=16)
    axs[3].set_ylabel('Field', fontsize=16)
    axs[6].set_ylabel('Mixed', fontsize=16)

    axs[6].set_xlabel('EfficientNet B0', fontsize=16)
    axs[7].set_xlabel('MobileNet V3s', fontsize=16)
    axs[8].set_xlabel('ResNet 34', fontsize=16)

    plt.tight_layout()
    plt.savefig('../fig/train_acc_loss.png', dpi=300)
    plt.show()


def cross_test_plot():
    predict_path = sorted(glob.glob('../data/cross_tested/*csv'))
    models = [path.split('/')[-1].split('_')[0] for path in predict_path]
    sources = [path.split('/')[-1].split('_')[1] for path in predict_path]
    targets = [path.split('/')[-1].split('_')[2][:-4] for path in predict_path]
    f = lambda x: x.split('/')[-2]
    accs = []
    for fn in predict_path:
        df = pd.read_csv(fn, index_col=0)
        df['true_label'] = df['img'].apply(f)
        acc = accuracy_score(df['true_label'], df['cla'])
        accs.append(acc)
    result = pd.DataFrame({'models': models, 'sources': sources, 'targets': targets, 'acc': accs})
    field = result[result['sources'] == 'field']
    lab = result[result['sources'] == 'lab']
    mixed = result[result['sources'] == 'mixed']

    from matplotlib.pyplot import bar
    plt.rcParams['ytick.labelsize'] = 12
    colors = {'lab': '#AE3D3A', 'mixed': '#3382A3', 'field': '#304E6C'}
    models_labels = ['EfficientNet B0', 'MobileNet V3 Small', 'ResNet34']

    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(15, 5), facecolor='w')
    # lab on field
    b1 = axs[0].bar(x=np.arange(3) - 0.1, height=lab['acc'].values[0::2], width=0.2, color=colors['field'],
                    label='Test on Field')
    b2 = axs[0].bar(x=np.arange(3) + 0.1, height=lab['acc'].values[1::2], width=0.2, color=colors['mixed'],
                    label='Test on Mixed')
    axs[0].set_ylim(0, 1)
    axs[0].set_ylabel('Accuracy', fontsize=14)
    axs[0].set_title('Train on Lab', fontsize=16)
    axs[0].set_xticks([0, 1.05, 2], labels=models_labels, fontsize=14)
    axs[0].xaxis.set_ticks_position('none')

    b3 = axs[1].bar(x=np.arange(3) - 0.1, height=mixed['acc'].values[0::2], width=0.2, color=colors['field'],
                    label='Test on Field')
    b4 = axs[1].bar(x=np.arange(3) + 0.1, height=mixed['acc'].values[1::2], width=0.2, color=colors['lab'],
                    label='Test on Lab')
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel('Accuracy', fontsize=14)
    axs[1].set_title('Train on Mixed', fontsize=16)
    axs[1].set_xticks([0, 1.05, 2], labels=models_labels, fontsize=14)
    axs[1].xaxis.set_ticks_position('none')
    b5 = axs[2].bar(x=np.arange(3) - 0.1, height=field['acc'].values[0::2], width=0.2, color=colors['lab'],
                    label='Test on '
                          'Lab')
    b6 = axs[2].bar(x=np.arange(3) + 0.1, height=field['acc'].values[1::2], width=0.2, color=colors['mixed'],
                    label='Test '
                          'on '
                          'Mixed')
    axs[2].set_ylim(0, 1)
    axs[2].set_ylabel('Accuracy', fontsize=14)
    axs[2].set_title('Train on Field', fontsize=16)

    axs[2].set_xticks([0, 1.05, 2], labels=models_labels, fontsize=14)
    axs[2].xaxis.set_ticks_position('none')

    legends = [bar([0], [0], color=colors['field'], label='Test on Field'),
               bar([0], [0], color=colors['mixed'], label='Test on Mixed'),
               bar([0], [0], color=colors['lab'], label='Test on Lab')]
    axs[2].legend(handles=legends, loc=0, frameon=False, fontsize=16)

    axs[0].bar_label(b1, label_type='edge', fmt="%.2f", fontsize=12)
    axs[0].bar_label(b2, label_type='edge', fmt="%.2f", fontsize=12)
    axs[1].bar_label(b3, label_type='edge', fmt="%.2f", fontsize=12)
    axs[1].annotate('0.98', xy=(0.2, 0.9), fontsize=12)
    axs[1].annotate('0.98', xy=(1.2, 0.9), fontsize=12)
    axs[1].annotate('0.99', xy=(1.8, 0.9), fontsize=12)
    axs[2].bar_label(b5, label_type='edge', fmt="%.2f", fontsize=12)
    axs[2].bar_label(b6, label_type='edge', fmt="%.2f", fontsize=12)

    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('../fig/cross_test_row.png', dpi=300)
    # plt.savefig('../fig/cross_test_row.tiff', dpi=300)
    plt.show()


def get_partial_data2frame():
    train_val_fns = sorted(glob.glob('../data/partial/*/output/*acc*csv'))
    val_accs = []
    for fn in train_val_fns:
        df = pd.read_csv(fn, index_col=0)
        val_accs.append(df.val_acc.max())
    test_fns = sorted(glob.glob('../data/partial/*/output/*predict*.csv'))
    test_acc = []
    for fn in test_fns:
        df = pd.read_csv(fn, index_col=0)
        df['true_label'] = df.img.apply(lambda x: x.split('/')[4])
        tsc = accuracy_score(df.true_label, df.cla)
        test_acc.append(tsc)
    models = ['Efficient', 'MobileNet', 'ResNet'] * 10
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    a = np.asarray(a).reshape(-1, 1)
    a = a.repeat(3, axis=1).flatten().tolist()
    df = pd.DataFrame.from_dict({'percent': a, 'models': models, 'val acc': val_accs, 'test acc': test_acc})

    field_csvs = sorted(glob.glob('../data/part_test_on_field/*/*csv'))
    field_acc = []
    for fn in field_csvs:
        df = pd.read_csv(fn, index_col=0)
        df['true_label'] = df.img.apply(lambda x: x.split('/')[3])
        fsc = accuracy_score(df.true_label, df.cla)
        field_acc.append(fsc)
    f_res = pd.DataFrame.from_dict({'percent': a, 'models': models, 'acc': field_acc})
    return df, f_res


def partial_valid_test_acc_plot():
    df, _ = get_partial_data2frame()
    colors = ['#F65429', '#EED364', '#3382A3']
    sns.set_palette(sns.color_palette(colors))
    p1 = sns.lineplot(x='percent', y='val acc', hue='models', data=df, marker='*', lw=2, ms=12, legend=True)
    sns.lineplot(x='percent', y='test acc', hue='models', data=df, marker='o', lw=1.5, ms=6, alpha=0.7, legend=False)
    handles, _ = p1.get_legend_handles_labels()
    l1 = plt.legend(handles=handles, labels=['EfficientNet B0', 'MobileNet V3 Small', 'ResNet 34'], frameon=False,
                    fontsize=12, bbox_to_anchor=[0.55, 0.5])
    custom_scatter = [plt.scatter(0, 0, marker='*', label='Validation Accuracy', color='k'),
                      plt.scatter(0, 0, marker='o', label='Test Accuracy', color='k')]
    l2 = plt.legend(handles=custom_scatter, fontsize=12, frameon=False, bbox_to_anchor=[0.55, 0.3])

    p1.add_artist(l1)
    p1.add_artist(l2)

    plt.xticks(ticks=np.arange(1, 11, 1), labels=['{}%'.format(i) for i in range(10, 110, 10)], fontsize=12)
    plt.xlabel('Percentage', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.savefig('../figs/lab_add2field_val_test_acc.png', dpi=300)
    plt.show()


def partial_test_on_filed_plot():
    _ , df = get_partial_data2frame()
    sns.lineplot(x=df['percent'], y=df['acc'], hue='models', linewidth=2,
                 marker='*', ms=12, data=df)
    plt.legend(labels=['EfficientNet B0', 'MobileNet V3 Small', 'ResNet 34'], frameon=False, fontsize=12, loc=[0.5, 0.25])
    plt.xticks(ticks=np.arange(1, 11, 1), labels=['{}%'.format(i) for i in range(10, 110, 10)], fontsize=12)
    plt.xlabel('Percentage', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.savefig('../figs/percent_test_on_field.png', dpi=300)
    plt.show()


def get_diseases2frame(criterion):
    total_test_fns = sorted(glob.glob(criterion))
    dfs = []
    for path in total_test_fns:
        condition = path.split('/')[0][2:]
        model = path.split('/')[-1].split('_')[0]
        df = pd.read_csv(path, index_col=0)
        df['true_label'] = df['img'].apply(lambda x: x.split('/')[3])
        matrix = confusion_matrix(df['true_label'], df['cla'], normalize='true')
        results = pd.DataFrame.from_dict({'diseases': df['true_label'].unique(), 'accuracy': matrix.diagonal()})
        results = results.sort_values(by='diseases', ascending=True)
        results['condition'] = condition
        results['model'] = model
        dfs.append(results)
    total = pd.concat(dfs, axis=0)
    total = total.reset_index(drop=True)
    return total


def get_diseases_individual2frame():
    limits_fns = sorted(glob.glob('./plant_individual/*/*/output/*predict_test.csv'))
    dfs = []
    for path in limits_fns:
        condition = path.split('/')[2]
        model = path.split('/')[-1].split('_')[0]
        df = pd.read_csv(path, index_col=0)
        df['true_label'] = df['img'].apply(lambda x: x.split('/')[5])
        matrix = confusion_matrix(df['true_label'], df['cla'], normalize='true')
        results = pd.DataFrame.from_dict({'diseases': df['true_label'].unique(), 'accuracy': matrix.diagonal()})
        results = results.sort_values(by='diseases', ascending=True)
        results['condition'] = condition
        results['model'] = model
        dfs.append(results)
    limit_df = pd.concat(dfs, axis=0)
    limit_df = limit_df.reset_index(drop=True)
    return limit_df


def diseases_acc_plot():
    total = get_diseases2frame(criterion='./*/output/*test.csv')
    limit_df = get_diseases_individual2frame()
    diseases = total.diseases.unique().tolist()
    diseases = [dis.replace('_', ' ') for dis in diseases]

    print(total.shape)
    print(limit_df.shape)

    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    fig, axs = plt.subplots(nrows=1, ncols=6, sharey=True, figsize=(8, 6), facecolor='w')
    i = 0
    for md in ['EfficientNet', 'MobileNetV3', 'ResNet']:
        tdf = total[total['model'] == md]
        ldf = limit_df[limit_df['model'] == md]
        tdata = tdf.accuracy.values.reshape(3, -1).T
        ldata = ldf.accuracy.values.reshape(3, -1).T
        p1 = axs[i].imshow(tdata, cmap='coolwarm_r')
        p2 = axs[i + 1].imshow(ldata, cmap='coolwarm_r')
        axs[i].set_xticks([0, 1, 2], ['Field', 'Mixed', 'Lab'], rotation=60)
        axs[i + 1].set_xticks([0, 1, 2], ['Field', 'Mixed', 'Lab'], rotation=60)
        axs[i].set_yticks(range(0, 14, 1), diseases)
        axs[i + 1].set_yticks(range(0, 14, 1), diseases)
        axs[i].set_title('a')
        axs[i + 1].set_title('b')
        i += 2

    plt.tight_layout()
    fig.text(0.3, 0.15, 'EfficientNet B0', fontsize=12)
    fig.text(0.48, 0.15, 'MobileNet V3 Small', fontsize=12)
    fig.text(0.71, 0.15, 'ResNet 34', fontsize=12)
    fig.colorbar(p2, ax=axs, shrink=0.6)

    plt.savefig('../figs/diseases_acc_change.png', dpi=300)
    plt.show()


def precision_recall_f1_plot(cla='precision'):
    """
    many data are impposible to make tables
    :return:
    """
    fns = sorted(glob.glob('../data/tables/?????_[!v]*'))
    dfs = []
    for fn in fns:
        model = fn.split('_')[2][:-4]
        cond = fn.split('_')[1]
        df = pd.read_csv(fn, index_col=0)
        df = df.iloc[:-3, :-1]
        df['conditions'] = cond
        df['models'] = model
        dfs.append(df)
    res = pd.concat(dfs, axis=00)

    colors =['#AE3D3A', '#3382A3', '#304E6C']

    plt.figure(figsize=(6, 6))
    s =sns.scatterplot(x=res.index, y=res[cla], hue='conditions', style='models', data=res, palette=colors)
    labels = [name.replace('_', ' ') for name in res.index.unique().tolist()]
    s.set_xticks(ticks=range(0, 14), labels=labels, rotation=90, fontsize=12)
    plt.legend(frameon=False, fontsize=12)
    plt.ylabel(cla.capitalize().replace('-', ' '), fontsize=12)
    plt.tight_layout()
    plt.savefig('../fig/total_'+cla+'.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    # data_statistics_bar_plot()
    # train_acc_loss_plot()
    # cross_test_plot()
    # partial_valid_test_acc_plot()
    # partial_test_on_filed_plot()
    # diseases_acc_plot()
    precision_recall_f1_plot(cla='f1-score')