
import os
import shutil
import torch

__all__ = ["save_checkpoint", "save_model"]

def save_checkpoint(state, out, show_epoch=False, is_best=False):
    if show_epoch == True:
        filename = os.path.join(out, 'checkpoint'+ state['epoch']+'.pth.tar')
    else:
        filename = os.path.join(out, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(out, 'model_best.pth.tar'))
        if(state["epoch"]<=15):
            shutil.copyfile(filename, os.path.join(out, 'model_best_foward.pth.tar'))


def save_model(state, out):
    filename = os.path.join(out, 'model_best.pth.tar')
    torch.save(state, filename)


