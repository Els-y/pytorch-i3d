# I3D SthSth

代码基于 [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d) 修改。

## 环境

- Python 3.x
- PyTorch 0.3
- numpy
- tensorboardX
- tqdm

## 数据准备

数据集下载 [Something-Something](https://20bn.com/datasets/something-something) 并解压，应有如下数据和标注文件：

```plain
.
├── 20bn-something-something-v2
│   ├── 1.webm
│   ├── 2.webm
│   └── ...
└── label
    ├── something-something-v2-labels.json
    ├── something-something-v2-test.json
    ├── something-something-v2-train.json
    └── something-something-v2-validation.json
```

## 训练

### 修改数据集路径

修改 `train_i3d.py` 文件 56 行左右的 `create_dataloader` 中数据集路径，其中：

- `split_file`: `something-something-v2-train.json`/`something-something-test.json`/`something-something-validation.json` 路径
- `label_file`: `something-something-v2-labels.json` 路径
- `webm_dir`: 存放解压后视频文件的目录 `20bn-something-something-v2` 路径

### 修改训练参数

参照 `train_i3d.py` 修改 `train.sh` 中对应参数以及可用 GPU。注意 `batch` 不能设大，目前试过 TITAN Xp 4 卡 `batch` 可以设置为 8，对应 `worker` 也设置为 8。

修改好后执行 `./train.sh` 即可开始训练。

### 训练结果

训练过程中每个 *epoch* 会保存一次模型，此外 *tensorboard* 中记录 *loss* 和 *accuracy* 的横坐标也为 *epoch*。

输出文件都存储在 `output` 目录中，并备份代码，按照训练开始时间划分目录:

```plain
output
├── sthsth_03-05_16:02:56
│   ├── codes
│   │   ├── pytorch_i3d.py
│   │   ├── sthsth_dataset.py
│   │   └── ...
│   ├── logs
│   │   └── events.out.tfevents.1583655045.club02
│   ├── train.sh
│   └── weights
│       ├── sthsth_1.pt
│       ├── sthsth_2.pt
│       ├── ...
│       └── sthsth_best.pt
└── sthsth_03-06_19:54:54
    └── ...
```

