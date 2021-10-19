from torch.utils.data import DataLoader
from .svhn import svhn_dataset

def get_dataset(args):
    f = globals().get(args.data_name+'_dataset')
    return f(args)

def get_dataloader(args):
    datasets = get_dataset(args)

    trainloader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=False)
    testloader = DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    dataloaders = {'train':trainloader,'test':testloader}

    return dataloaders
