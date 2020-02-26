import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from train_i3d import create_dataloaders
from pytorch_i3d import InceptionI3d
from utils import topk_corrects


def main(mode, batch_size, weight_path):
    print('prepare dataset...')
    dataloaders = create_dataloaders(mode, batch_size=batch_size)

    print('load model...')
    model = create_model(mode, weight_path)

    print('start testing...')
    test(model, dataloaders['train'])


def create_model(mode, weight_path):
    i3d = InceptionI3d(400, in_channels=3)
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
    for data in tqdm(dataloader):
        inputs, labels = data

        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        logits = model(inputs)
        corrects += topk_corrects(logits.cpu().data, labels.cpu().data, topk)
        nums += inputs.size(0)

    acc = corrects / nums
    for i, k in enumerate(topk):
        print('top{} acc: {:.2f}% ({}/{})'.format(k, 100.0 * acc[i], corrects[i], nums))


if __name__ == '__main__':
    main('rgb', 1, 'output/sthsth_1.pt')
