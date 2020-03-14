import os
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from train_i3d import create_dataloaders
from pytorch_i3d import InceptionI3d
from resnet3d import resnet50
from utils import topk_corrects


def main(mode, net, phase, weight, batch_size, num_workers):
    print('prepare dataset...')
    dataloaders = create_dataloaders(mode, batch_size=batch_size, num_workers=num_workers)

    print('load model...')
    model = create_model(mode, net, weight)

    print('start testing...')
    test(model, dataloaders[phase])


def create_model(mode, net, weight_path):
    if net == 'inception':
        i3d = InceptionI3d(400, in_channels=3)
    else:
        i3d = resnet50()
    i3d.replace_logits(174)

    params = OrderedDict()
    for key, values in torch.load(weight_path).items():
        params[key[7:]] = values
    i3d.load_state_dict(params)

    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    return i3d


def test(model, dataloader):
    topk = (1, 5)
    corrects = torch.zeros(len(topk))
    nums = 0
    progress_bar = tqdm(total=len(dataloader), desc='test')
    for data in dataloader:
        inputs, labels = data

        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        logits = model(inputs)
        corrects += topk_corrects(logits.cpu().data, labels.cpu().data, topk)
        nums += inputs.size(0)

        acc = corrects / nums
        disp_dict = {
            'Acc@1': '{:.1f}% ({:d})'.format(100 * acc[0], corrects.long()[0]),
            'Acc@5': '{:.1f}% ({:d})'.format(100 * acc[1], corrects.long()[1]),
            'Size': '{:d}'.format(nums)}
        progress_bar.set_postfix(disp_dict)
        progress_bar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='rgb', choices=['rgb'], help='mode')
    parser.add_argument('--net', type=str, default='resnet50', choices=['resnet50', 'inception'], help='net architecture')
    parser.add_argument('--phase', type=str, default='val', choices=['train', 'val'], help='phase')
    parser.add_argument('--weight', type=str, required=True, help='weight path')
    parser.add_argument('--batch', type=int, default=4, help='num of batch')
    parser.add_argument('--worker', type=int, default=4, help='num of workers')
    args = parser.parse_args()
    print(args)

    main(mode=args.mode,
         net=args.net,
         phase=args.phase,
         weight=args.weight,
         batch_size=args.batch,
         num_workers=args.worker)

