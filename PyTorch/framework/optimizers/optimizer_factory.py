import torch.optim as optim

def adam(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001, amsgrad=False):
    if isinstance(betas, str):
        betas = eval(betas)
    return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay,amsgrad=amsgrad)

def sgd(parameters, lr=0.1, momentum=0.9, weight_decay=0.0001):
    return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

def get_optimizer(args, parameters):
    f = globals().get(args.optimizer)
    return f(parameters, lr=args.lr, weight_decay=args.weight_decay)
    
def get_q_optimizer(args, parameters):
    f = globals().get(args.quant_optimizer)
    return f(parameters,lr=args.quant_lr, weight_decay=args.quant_weight_decay)
