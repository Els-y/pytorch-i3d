import torch

def topk_corrects(logits, labels, topk=(1,)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, 1)
    _, gt = labels.topk(1, 1)
    correct = pred.eq(gt.expand_as(pred))
    ret = torch.zeros(len(topk))
    for i, k in enumerate(topk):
        ret[i] = correct[:, :k].sum()
    return ret

