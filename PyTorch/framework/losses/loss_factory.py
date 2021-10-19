import torch

def cross_entropy(reduction='mean'):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)

    def loss_fn(outputs, labels):
        gt_loss = cross_entropy_fn(outputs, labels)
        return gt_loss

    return loss_fn

def get_loss(args):
    f = globals().get(args.loss)
    return f()
