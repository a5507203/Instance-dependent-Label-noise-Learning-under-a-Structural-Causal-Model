__all__ = ["adjust_learning_rate"]

def adjust_learning_rate(optimizer, epoch, base_lr, ajust_period=70):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = base_lr * (0.1 ** (epoch // ajust_period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



