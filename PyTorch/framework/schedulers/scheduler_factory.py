import torch.optim.lr_scheduler as lr_scheduler

def step(optimizer, step_size=90, gamma=0.1):
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

def multi_step(optimizer, milestones=[90], gamma=0.1):
# def multi_step(optimizer, milestones=[250,280], gamma=0.1):
    if isinstance(milestones, str):
        milestones = eval(milestones)
    return lr_scheduler.MultiStepLR(optimizer, milestones=milestones,gamma=gamma)

def exponential(optimizer, gamma=0.995):
    return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

def reduce_lr_on_plateau(optimizer, mode='min', factor=0.1, patience=10,threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0):
    return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                        threshold=threshold, threshold_mode=threshold_mode,
                                        cooldown=cooldown, min_lr=min_lr)

def cosine(optimizer, T_max=100, eta_min=0.00001):
    print('cosine annealing, T_max: {}, eta_min: {}'.format(T_max+1, eta_min))
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max+1, eta_min=eta_min)

def get_scheduler(args, optimizer):
    func = globals().get(args.scheduler)
    return func(optimizer)

def get_q_scheduler(args, optimizer):
    func = globals().get(args.scheduler)
    return func(optimizer)
