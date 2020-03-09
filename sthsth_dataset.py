import os
import json
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data_util


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def resize_smaller_side(bgr, min_side):
    h, w, _ = bgr.shape
    if min(h, w) < min_side:
        d = min_side - min(h, w)
        sc = 1 + float(d) / min(h, w)
        bgr = cv2.resize(bgr, dsize=(0, 0), fx=sc, fy=sc)
    return bgr


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


class SthSthDataset(data_util.Dataset):

    def __init__(self, phase, split_file, label_file, webm_dir,
                 mode, transforms=None):
        self.phase = phase
        self.split_file = split_file
        self.label_file = label_file
        self.webm_dir = webm_dir
        self.transforms = transforms
        self.class_num = 174
        self.frame_num = 64
        self.min_side = 256
        assert phase in ['train', 'val', 'test']
        if phase == 'train':
            self.data = load_json(self.split_file)
        else:
            self.data = load_json(self.split_file)
        self.label_map = load_json(self.label_file)

    def get_rgb_frames(self, index):
        webm_id = self.data[index]['id']
        webm_file = os.path.join(self.webm_dir, webm_id + '.webm')
        webm = cv2.VideoCapture(webm_file)
        assert webm.isOpened()

        frames = []
        while True:
            ok, frame_bgr = webm.read()
            if not ok:
                if len(frames) >= self.frame_num:
                    break
                else:
                    webm.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            frame_bgr = resize_smaller_side(frame_bgr, self.min_side)
            frame_rgb = frame_bgr[:, :, [2, 1, 0]]
            frame_rgb = (frame_rgb / 255.0) * 2 - 1
            frames.append(frame_rgb.astype(np.float32))

        start_idx = random.randint(0, len(frames) - self.frame_num)
        data = frames[start_idx:start_idx + self.frame_num]
        return np.asarray(data)

    def get_label(self, index):
        if self.phase == 'test':
            return -1

        template = self.data[index]['template'].replace('[', '').replace(']', '')
        label = int(self.label_map[template])

        return label

    def __getitem__(self, index):
        label = self.get_label(index)
        rgb_inputs = self.get_rgb_frames(index)

        if self.transforms is not None:
            rgb_inputs = self.transforms(rgb_inputs)

        return video_to_tensor(rgb_inputs), label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    import videotransforms
    from torchvision import transforms

    # tfs = transforms.Compose([videotransforms.RandomCrop(224),
    #                          videotransforms.RandomHorizontalFlip()])
    tfs = transforms.Compose([videotransforms.CenterCrop(224)])
    dataset = SthSthDataset(phase='val',
                            split_file='/data3/yejiaquan/data/sthsth/label/something-something-v2-validation.json',
                            label_file='/data3/yejiaquan/data/sthsth/label/something-something-v2-labels.json',
                            webm_dir='/data3/yejiaquan/data/sthsth/20bn-something-something-v2',
                            mode='rgb',
                            transforms=tfs)

    for i in range(5):
        inputs, label = dataset[i]
        inputs = inputs.numpy().transpose([1, 2, 3, 0])
        inputs = (inputs + 1) / 2 * 255.0
        inputs = inputs.astype(np.uint8)
        inputs = inputs[:, :, :, [2, 1, 0]]
        folder = 'center/{}'.format(i)
        os.makedirs(folder, exist_ok=True)
        for j in range(64):
            cv2.imwrite(os.path.join(folder, '{}.jpg'.format(j)), inputs[j])
