import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

import videotransforms
from pytorch_i3d import InceptionI3d

from sthsth_dataset import SthSthDataset
from utils import topk_corrects


def main(mode, num_epochs, batch_size, lr, save_prefix):
    print('prepare dataset...')
    dataloaders = create_dataloaders(mode, batch_size=batch_size)

    print('load model...')
    model = create_model(mode)

    print('prepare optimizer...')
    optimizer, scheduler = create_optimizer(model, lr)

    writer = SummaryWriter('mgpu')

    print('start training...')
    train(num_epochs, model, dataloaders, optimizer, scheduler, save_prefix, writer)

    writer.close()

def create_dataloaders(mode, batch_size=4, num_workers=4):
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip()])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    train_dataset = SthSthDataset(phase='train',
                            split_file='/data3/yejiaquan/data/sthsth/label/something-something-v2-train.json',
                            label_file='/data3/yejiaquan/data/sthsth/label/something-something-v2-labels.json',
                            webm_dir='/data3/yejiaquan/data/sthsth/20bn-something-something-v2',
                            mode=mode,
                            transforms=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

    val_dataset = SthSthDataset(phase='val',
                                split_file='/data3/yejiaquan/data/sthsth/label/something-something-v2-validation.json',
                                label_file='/data3/yejiaquan/data/sthsth/label/something-something-v2-labels.json',
                                webm_dir='/data3/yejiaquan/data/sthsth/20bn-something-something-v2',
                                mode=mode,
                                transforms=test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    return dataloaders


def create_model(mode, num_classes=174):
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(num_classes)
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    return i3d


def create_optimizer(model, lr, momentum=0.9):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 50])
    return optimizer, scheduler


def train(num_epochs, model, dataloaders, optimizer, scheduler, save_prefix, writer):
    progress_bar = {
        'epoch': tqdm(total=num_epochs, desc='epoch'),
        'batch': tqdm(desc='batch')}

    topk = (1, 5)
    best_acc = 0.0
    epoch_disp = {
        'tLoss': 0,
        'vLoss': 0,
        'tAcc@1,5': '{:.2f}%,{:.2f}%'.format(0., 0.),
        'vAcc@1,5': '{:.2f}%,{:.2f}%'.format(0., 0.),
    }
    for epoch in range(1, num_epochs + 1):
        progress_bar['epoch'].update()
        scheduler.step()

        for phase in ['train', 'val']:
            progress_bar['batch'].reset(len(dataloaders[phase]))

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = torch.zeros(len(topk))
            running_nums = 0
            for data in dataloaders[phase]:
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())  # (B x C x T x H x W)
                labels = Variable(labels.cuda())  # (B x 174)

                optimizer.zero_grad()

                logits = model(inputs)
                loss = F.binary_cross_entropy_with_logits(logits, labels)

                # _, preds = torch.max(logits.cpu().data, 1)
                # _, gt = torch.max(labels.cpu().data, 1)

                running_loss += loss.data[0]
                running_nums += inputs.size(0)
                running_corrects += topk_corrects(logits.cpu().data, labels.cpu().data, topk)
                # running_corrects += torch.sum(preds == gt)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_acc = running_corrects / running_nums
                batch_disp = {
                    'phase': phase,
                    'loss': loss.data[0],
                    'acc@1': '{:.1f}% ({})'.format(100. * running_acc[0], int(running_corrects[0])),
                    'acc@5': '{:.1f}% ({})'.format(100. * running_acc[1], int(running_corrects[1])),
                    'size': running_nums
                }
                progress_bar['batch'].set_postfix(batch_disp)
                progress_bar['batch'].update()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            if phase == 'train':
                epoch_disp['tLoss'] = epoch_loss
                epoch_disp['tAcc@1,5'] = '{:.1f}%,{:.1f}%'.format(100. * epoch_acc[0], 100. * epoch_acc[1])
            else:
                epoch_disp['vLoss'] = epoch_loss
                epoch_disp['vAcc@1,5'] = '{:.1f}%,{:.1f}%'.format(100. * epoch_acc[0], 100. * epoch_acc[1])
            progress_bar['epoch'].set_postfix(epoch_disp)

            writer.add_scalar('{}/loss'.format(phase), loss.data[0], epoch)
            writer.add_scalar('{}/correct/top1'.format(phase), running_corrects[0], epoch)
            writer.add_scalar('{}/correct/top5'.format(phase), running_corrects[1], epoch)
            writer.add_scalar('{}/acc/top1'.format(phase), epoch_acc[0], epoch)
            writer.add_scalar('{}/acc/top5'.format(phase), epoch_acc[1], epoch)
            writer.add_scalar('{}/size'.format(phase), len(dataloaders[phase].dataset))

            # top1 acc
            if phase == 'val' and epoch_acc[0] > best_acc:
                best_acc = epoch_acc[0]
                torch.save(model.state_dict(), '{}_best.pt'.format(save_prefix))

        torch.save(model.state_dict(), '{}_{}.pt'.format(save_prefix, epoch))


if __name__ == '__main__':
    main(mode='rgb', num_epochs=70, batch_size=4, lr=0.001, save_prefix='output/sthsth_4gpu')
