# -*- coding: utf-8 -*-
# @Time    : 2022/10/24 下午9:46
# @Author  : weather
# @Software: PyCharm

import numpy as np
import random
import torch
from torch import nn as nn
import time
import torch.optim as optim
from collections import defaultdict
import pandas as pd
from torch.optim import lr_scheduler
import copy
from model import set_model
from datasets import load_datasets, data_transforms
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using {} device'.format(device))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmasrk = False


def train_model(model, num_epochs, dataset_sizes, dataloaders, optimizer, criterion, scheduler, outdir):

    set_seed()
    model_name = model._get_name()
    model.to(device)

    print(model_name + ' is trainning now')
    print("-" * 20)

    save_weights = defaultdict(list)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                save_weights['train_acc'].append(epoch_acc.data.cpu().numpy())
                save_weights['train_loss'].append(epoch_loss)
            if phase == 'valid':
                save_weights['val_acc'].append(epoch_acc.data.cpu().numpy())
                save_weights['val_loss'].append(epoch_loss)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                patience = 0
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model_path = outdir + model_name + '.pth'
                # load best model weights
                model.load_state_dict(best_model_wts)
                torch.save(model.state_dict(), model_path)
            elif phase == 'valid' and epoch_acc <= best_acc:
                patience += 1
                print("Counter {} of 5".format(patience))

        if patience > 4:
            print("Early stopping with best_acc:{:.4f} ".format(best_acc.item()))
            break

    df = pd.DataFrame.from_dict(save_weights)
    outname_csv = outdir + model_name + '_acc_loss.csv'
    df.to_csv(outname_csv)
    time_elapsed = time.time() - since
    txtfile = outdir + "train_log.txt"
    with open(txtfile, 'a+') as f:
        f.write(model_name)
        f.write("    ")
        f.write(str(time_elapsed))
        f.write('\n')
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input folder of images, root dir including each class folder')
    parser.add_argument('-o', '--output', type=str, help='output folder')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='epochs for training')
    parser.add_argument('-b', '--batchsize', type=int, default=64, help='batch_size of each epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function, ')
    parser.add_argument('-m', '--model', type=str, help='CNN models, only use effi0(Efficient Net b0),'
                                                        ' mobilev3(Mobile Net V3s) and res34(ResNet34)')
    parser.add_argument('--optim', type=str, default='sgd',  help='optimizer of weights, use sgd or adam')
    args = parser.parse_args()
    return args


def main(args):
    data_transform = data_transforms()
    dataset_sizes, dataloaders, class_names = load_datasets(args.input, data_transform, args.batchsize)
    num_cls = len(class_names.keys())
    model = set_model(num_cls, args.model)
    if args.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplemented

    if args.optim == 'sgd':
        optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    elif args.optim == 'adam':
        optimizer_ft = optim.adam.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    else:
        raise NotImplemented

    train_model(model, args.epochs, dataset_sizes, dataloaders, optimizer_ft, criterion, exp_lr_scheduler, args.output)


if __name__ == "__main__":
    args = arg_parser()
    main(args)
    # from torchvision.models import resnet18
    # optimizer_ft = optim.SGD(resnet18(pretrained=False).parameters(), lr=0.001, momentum=0.9)
    # lrs = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    # for i in range(20):
    #     optimizer_ft.step()
    #     lrs.step()
    #     print(lrs.get_last_lr())
