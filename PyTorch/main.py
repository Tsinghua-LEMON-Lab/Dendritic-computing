import os
import time

import tqdm
import torch
import argparse
import warnings
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from framework.datasets import get_dataloader
from framework.models import get_model
from framework.losses import get_loss
from framework.optimizers import get_optimizer
from framework.schedulers import get_scheduler
from framework.utils import AverageMeter, accuracy, count_parameters, load_checkpoint, param_extract, save_checkpoint

warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_single_epoch(model, dataloader, criterion, optimizer, epoch, writer, postfix_dict):
    model.train()
    total_step = len(dataloader)
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    losses = AverageMeter()
    top1 = AverageMeter()
    log_dict={}
    for i, (imgs, labels) in tbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        pred_dict = model(imgs)
        # print(pred_dict.sum())
        loss = criterion(pred_dict, labels)

        loss.backward()

        losses.update(loss, labels.size(0))
        log_dict['loss'] = losses.avg.item()

        prec1, _ = accuracy(pred_dict.data, labels.data, topk=(1, 5))
        top1.update(prec1[0], labels.size(0))
        log_dict['accuracy'] = top1.avg.item()

        optimizer.step()
        f_epoch = epoch + i / total_step
        log_dict['lr'] = optimizer.param_groups[0]['lr']
        for key, value in log_dict.items():
            postfix_dict['train/{}'.format(key)] = value

        desc = '{:5s}'.format('train') + ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)

    # tensorboard
    if writer is not None:
        for key, value in log_dict.items():
            writer.add_scalar('train/{}'.format(key), value, epoch)

def evaluate_single_epoch(model, dataloader, criterion, epoch, writer, postfix_dict):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    with torch.no_grad():
        total_step = len(dataloader)
        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)

        for i, (imgs, labels) in tbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            pred_dict = model(imgs)
            train_loss = criterion(pred_dict, labels)
            prec1, prec5 = accuracy(pred_dict.data, labels.data, topk=(1,5))

            losses.update(train_loss.item(), labels.size(0))
            top1.update(prec1[0], labels.size(0))
            top5.update(prec5[0], labels.size(0))

            ## Logging
            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format('test') + ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
            tbar.set_description(desc)
            tbar.set_postfix(**postfix_dict)

        ## logging
        log_dict = {}
        log_dict['loss'] = losses.avg
        log_dict['accuracy'] = top1.avg.item()
        # log_dict['top5'] = top5.avg.item()

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('{}/{}'.format('test', key), value, epoch)
            postfix_dict['{}/{}'.format('test', key)] = value

        return log_dict['loss'], log_dict['accuracy']

def evaluate_test(args, model, dataloader, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (imgs, labels) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            imgs = imgs.to(device)
            labels = labels.to(device)

            pred_dict = model(imgs)
            train_loss = criterion(pred_dict, labels)
            prec1, prec5 = accuracy(pred_dict.data, labels.data, topk=(1,5))

            losses.update(train_loss.item(), labels.size(0))
            top1.update(prec1[0], labels.size(0))
            top5.update(prec5[0], labels.size(0))
    acc_path = os.path.join(args.save_path, 'acc')
    np.save(acc_path, np.array([top1.avg.item()]))
    print('test losses:%.3f, top1:%.3f'%(losses.avg, top1.avg.item()))

def train(args, model, dataloaders, criterion, optimizer, scheduler):
    writer = SummaryWriter(log_dir=args.save_path)
    num_epochs = args.num_epochs

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    postfix_dict = {'train/lr': 0.0,
                    'train/loss': 0.0,
                    'train/accuracy': 0.0,
                    'test/accuracy':0.0,
                    'test/loss':0.0}
    best_accuracy = 0.
    best_epoch = 0
    for epoch in range(num_epochs):
        # train phase
        train_single_epoch(model, dataloaders['train'], criterion, optimizer, epoch, writer, postfix_dict)
        # test phase
        loss, top1 = evaluate_single_epoch(model, dataloaders['test'], criterion, epoch, writer, postfix_dict)

        scheduler.step()

        if best_accuracy < top1:
            best_accuracy = top1
            save_checkpoint(args, model, optimizer, scheduler, is_best=True)
            best_epoch = epoch

    save_checkpoint(args, model, optimizer, scheduler, is_best=False)

    print('best_accuracy:%.3f at epoch %d , last_accuracy:%.3f'%(best_accuracy, best_epoch, top1))

def run(args):

    dataloaders = get_dataloader(args)
    print("used data: %s" % (args.data_name))
    model = get_model(args).to(device)
    print("The number of train parameters : %d" % count_parameters(model))
    criterion = get_loss(args)
    print("used loss function : %s" % (args.loss))

    # Loading the full-precision model
    if args.pretrained_path:
        model = load_checkpoint(args, model).to(device)

    param = param_extract(model)
    optimizers = get_optimizer(args, param)
    schedulers = get_scheduler(args, optimizers)

    if args.eval:
        evaluate_test(args, model, dataloaders['test'], criterion)
        return
    else:
        train(args, model, dataloaders, criterion, optimizers, schedulers)

def parse_args():
    parser = argparse.ArgumentParser(description='Device precision parameter argsuration')
    parser.add_argument('--data_root', default='data', type=str,help='train dataset root path')
    parser.add_argument('--data_name', default='svhn', type=str,help='train dataset name')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_workers', default=0, type=int, help='number workers')
    parser.add_argument('--num_epochs', default=100, type=int, help='number epochs')

    parser.add_argument('--eval', default=False, type=bool, help='Only evaluation')
    parser.add_argument('--model_name', default='ann', type=str, help='model name')
    parser.add_argument('--pretrained_path', default='', type=str, help='model pretrained path')
    parser.add_argument('--save_root', default='results', type=str, help='The root directory that holds all output')
    parser.add_argument('--loss', default='cross_entropy', type=str, help='Loss function')

    parser.add_argument('--scheduler', default='multi_step', type=str, help='Learning rate attenuation strategy')
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')

    args = parser.parse_args()

    args.save_path = os.path.join(args.save_root, args.model_name, args.data_name)
    os.makedirs(args.save_path,exist_ok=True)

    if not args.eval:
        hyperparameter_writer = open(os.path.join(args.save_path,'hyperparameter.txt'), 'w')
        hyperparameter_writer.write('{}\n'.format({k: v for k, v in args._get_kwargs()}))
        hyperparameter_writer.flush()
        hyperparameter_writer.close()
    return args

if __name__ == '__main__':
    begin = time.time()
    args = parse_args()
    run(args)
    end = time.time()-begin
    print('success!, runing %dh-%dm-%ds'% (end // 3600, end % 3600 // 60, round(end % 60)))
