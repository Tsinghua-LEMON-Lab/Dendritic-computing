import os
import shutil
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def param_extract(model):

    model_params = []
    model_name = []
    for name, params in model.named_parameters():
        model_name.append(name)
        if 'wgt_alpha' in name:
            pass
        elif 'wgt_thet' in name:
            pass
        elif 'uW' in name or 'lW' in name or 'init' in name or 'beta' in name:#针对daq 但和原码并不相同
            pass
        else:
            model_params += [{'params': [params]}]

    return model_params

def save_checkpoint(args, model, optimizers, schedulers, is_best):

    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    weights_dict = {
      'state_dict': state_dict,
      'optimizer_dict': optimizers.state_dict(),
      'scheduler_dict': schedulers.state_dict(),
    }

    checkpoint_dir = os.path.join(args.save_path, 'checkpoint')
    os.makedirs(checkpoint_dir,exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'last.pth')
    torch.save(weights_dict, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(checkpoint_dir, 'best.pth'))

def load_checkpoint(args, model):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.pretrained_path)['state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('load the pretrained model')
    return model
