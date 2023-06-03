import torch
from flash.core.optimizers import LARS

# helper
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# optmizer function
def return_optimizer(optimizer):
    if optimizer == 'AdamW':
        return torch.optim.AdamW
    if optimizer == 'LARS':
        return LARS

def return_lr_scheduler(lr_scheduler):
    if lr_scheduler == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau
    if lr_scheduler == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR