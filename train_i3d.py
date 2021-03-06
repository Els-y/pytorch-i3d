import os
import sys
import argparse
import time
import shutil

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
from resnet3d import resnet50

from sthsth_dataset import SthSthDataset
from utils import topk_corrects


def main(mode, net, num_epochs, batch_size, lr, optimizer_type, pretrained, num_workers, output_dir, save_prefix):
    weight_storage = os.path.join(output_dir, 'weights')
    log_storage = os.path.join(output_dir, 'logs')
    os.makedirs(weight_storage, exist_ok=True)
    os.makedirs(log_storage, exist_ok=True)

    print('prepare dataset...')
    dataloaders = create_dataloaders(mode, batch_size=batch_size, num_workers=num_workers)
    print('train size: {}, train iter: {}, val size: {}, val iter: {}'.format(
        len(dataloaders['train'].dataset), len(dataloaders['train']),
        len(dataloaders['val'].dataset), len(dataloaders['val'])))

    print('load model...')
    model = create_model(mode, net, pretrained)

    print('prepare optimizer...')
    optimizer, scheduler = create_optimizer(model, optimizer_type, lr)

    writer = SummaryWriter(log_storage)

    print('start training...')
    train(num_epochs, model, dataloaders, optimizer, scheduler,
          os.path.join(weight_storage, save_prefix), writer)

    writer.close()


def create_dataloaders(mode, batch_size=4, num_workers=1):
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


def create_model(mode, net, pretrained=True, num_classes=174):
    if mode == 'flow':
        raise ValueError('unsupported mode.')

    # rgb mode
    if net == 'inception':
        i3d = InceptionI3d(400, in_channels=3)
        if pretrained:
            i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    elif net == 'resnet50':
        i3d = resnet50()
        if pretrained:
            i3d.load_state_dict(torch.load('models/resnet50_3d.pth'))
    else:
        raise ValueError('unknown net.')

    i3d.replace_logits(num_classes)
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    return i3d


def create_optimizer(model, optimizer_type, lr):
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [70])
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
    batch_x = {'train': 1, 'val': 1}
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
                labels = Variable(labels.cuda())  # (B x C), C = 174

                optimizer.zero_grad()

                logits = model(inputs)
                loss = F.cross_entropy(logits, labels)

                # _, preds = torch.max(logits.cpu().data, 1)
                # _, gt = torch.max(labels.cpu().data, 1)

                running_loss += loss.data[0] * inputs.size(0)
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

                writer.add_scalar('{}/iter/loss'.format(phase), loss.data[0], batch_x[phase])
                writer.add_scalar('{}/iter/running_acc/top1'.format(phase), running_acc[0], batch_x[phase])
                writer.add_scalar('{}/iter/running_acc/top5'.format(phase), running_acc[1], batch_x[phase])
                batch_x[phase] += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            if phase == 'train':
                epoch_disp['tLoss'] = epoch_loss
                epoch_disp['tAcc@1,5'] = '{:.1f}%,{:.1f}%'.format(100. * epoch_acc[0], 100. * epoch_acc[1])
            else:
                epoch_disp['vLoss'] = epoch_loss
                epoch_disp['vAcc@1,5'] = '{:.1f}%,{:.1f}%'.format(100. * epoch_acc[0], 100. * epoch_acc[1])
            progress_bar['epoch'].set_postfix(epoch_disp)

            writer.add_scalar('{}/loss'.format(phase), epoch_loss, epoch)
            writer.add_scalar('{}/acc/top1'.format(phase), epoch_acc[0], epoch)
            writer.add_scalar('{}/acc/top5'.format(phase), epoch_acc[1], epoch)
            writer.add_scalar('{}/size'.format(phase), len(dataloaders[phase].dataset))

            # top1 acc
            if phase == 'val' and epoch_acc[0] > best_acc:
                best_acc = epoch_acc[0]
                torch.save(model.state_dict(), '{}_best.pt'.format(save_prefix))

        torch.save(model.state_dict(), '{}_{}.pt'.format(save_prefix, epoch))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='rgb', choices=['rgb'], help='mode')
    parser.add_argument('--net', type=str, default='resnet50', choices=['resnet50', 'inception'], help='net architecture')
    parser.add_argument('--epoch', type=int, default=80, help='num of epoch')
    parser.add_argument('--batch', type=int, default=4, help='num of batch')
    parser.add_argument('--lr', type=float, default=0.00125, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer')
    parser.add_argument('--pretrained', type=str2bool, default=True, help='if use pretrained model')
    parser.add_argument('--worker', type=int, default=4, help='num of workers')
    parser.add_argument('--output', type=str, default='output/sthsth_{}'.format(time.strftime('%m-%d_%H:%M:%S')), help='output dir')
    parser.add_argument('--prefix', type=str, default='sthsth')
    args = parser.parse_args()
    print(args)

    main(mode=args.mode,
         net=args.net,
         num_epochs=args.epoch,
         batch_size=args.batch,
         lr=args.lr,
         optimizer_type=args.optimizer,
         pretrained=args.pretrained,
         num_workers=args.worker,
         output_dir=args.output,
         save_prefix=args.prefix)

