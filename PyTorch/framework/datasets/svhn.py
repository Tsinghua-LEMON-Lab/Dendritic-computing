import torchvision

def svhn_dataset(args):

    # Data augmentation
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.SVHN(root=args.data_root, split='train', download=True, transform=train_transform)

    testset = torchvision.datasets.SVHN(root=args.data_root, split='test', download=True, transform=test_transform)

    return {"train":trainset, "test":testset}
