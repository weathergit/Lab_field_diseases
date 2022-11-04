# -*- coding: utf-8 -*-
# @Time    : 2022/10/31 下午3:19
# @Author  : weather
# @Software: PyCharm

import glob
import os
import pandas as pd
from sklearn.metrics import classification_report


def get_val_acc():
    valid_paths = sorted(glob.glob('../data/*/output/*acc*'))
    val_accs = []
    models = []
    conditions = []
    for path in valid_paths:
        model = path.split('/')[-1].split('_')[0]
        cond = path.split('/')[2]
        df = pd.read_csv(path, index_col=0)
        val_max = df['val_acc'].max()
        val_accs.append(val_max)
        models.append(model)
        conditions.append(cond)
    results = pd.DataFrame.from_dict({'condition': conditions, 'models': models, 'acc': val_accs})
    results.to_csv('../data/tables/total_valid.csv',index=None)
    # print(results)


def get_test_report():
    test_paths = sorted(glob.glob('../data/*/output/*test*'))
    for path in test_paths:
        model = path.split('/')[-1].split('_')[0]
        cond = path.split('/')[2]
        df = pd.read_csv(path, index_col=0)
        df['label'] = df['img'].apply(lambda x: x.split('/')[3])
        res = classification_report(df['label'], df['cla'], output_dict=True)
        out = pd.DataFrame.from_dict(res).T
        # print(cond, model)
        # print(out)
        out.to_csv('../data/tables/total_'+cond+'_'+model+'.csv')


if __name__ == "__main__":
    # get_val_acc()
    get_test_report()