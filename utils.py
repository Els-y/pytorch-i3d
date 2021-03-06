import torch

def topk_corrects(logits, labels, topk=(1,)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, 1)
    pred = pred.t()
    correct = pred.eq(labels.expand_as(pred))
    ret = torch.zeros(len(topk))
    for i, k in enumerate(topk):
        ret[i] = correct[:k, :].sum()
    return ret


if __name__ == '__main__':
    logits = torch.FloatTensor([[1,2,3,4],[3,4,1,2],[4,3,2,1]])
    # labels = torch.LongTensor([[0,0,1,0],[0,1,0,0],[0,0,1,0]])
    labels = torch.LongTensor([2,1,2])
    ret = topk_corrects(logits, labels, (1,2,3,4))
    print(ret)
