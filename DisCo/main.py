#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import glob
from torch.cuda.amp import autocast, GradScaler

import moco.loader
import moco.builder_kq_mse_largeembedding_2048
#import moco.builder_kq_mse_largeembedding_2048_spatial
#from models.model import create_model, load_model, save_model
#from moco.datasets.jiaotong_plate import JiaoTongPlate

from models.efficientnet import efficientnet_b0
from models.efficientnet import efficientnet_b1
from models.mobilenetv3 import mobilenetv3_large_100
from models.resnet import resnet18, resnet34
from models.swav_resnet50 import resnet50w2
from models.swav_resnet50 import resnet50 as swav_resnet50

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names.append("efficientb0")
model_names.append("efficientb1")
model_names.append("mobilenetv3")
model_names.append("resnet18")
model_names.append("resnet34")
model_names.append("resnet50w2")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-ckpt', action='store_true',
                    help='whether or not to resume from ./ckpt')
parser.add_argument('--teacher_arch', default='resnet50', type=str,
                    help='teacher architecture')
parser.add_argument('--teacher', default='', type=str, metavar='PATH',
                    help='path to teacher checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--nmb_prototypes', default=0, type=int,
                    help='num prototype')
parser.add_argument('--only-mse', action='store_true',
                    help='only use mse loss')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print(f'Number of GPUs: {ngpus_per_node}')
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    ## ---- create backbone ----
    #base_arch = 'dlapruneprune_34'
    #base_heads = {'hm': 4, 'wh': 8, 'reg': 2}
    #base_head_conv = 256
    #base_model = create_model(base_arch, base_heads, base_head_conv)

    print("=> creating model '{}'".format(args.teacher_arch))
    if args.teacher_arch == 'resnet50w2':
        teacher_model = resnet50w2
    elif args.teacher_arch in ['resnet50', 'resnet101', 'resnet152']:
        teacher_model = models.__dict__[args.teacher_arch]
    elif args.teacher_arch in ['SWAVresnet50', 'DCresnet50', 'SELAresnet50']:
        teacher_model = swav_resnet50
    else:
        print('Error')
        sys.exit(-1)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == "efficientb0":
        model = efficientnet_b0
    elif args.arch == "efficientb1":
        model = efficientnet_b1
    elif args.arch == "mobilenetv3":
        model = mobilenetv3_large_100
    elif args.arch == "resnet18":
        # model = models.__dict__[args.arch]#resnet18(pretrained=False)
        model = resnet18
    elif args.arch == "resnet34":
        # model = models.__dict__[args.arch]#resnet18(pretrained=False)
        model = resnet34
    else:
        model = models.__dict__[args.arch]

    model = moco.builder_kq_mse_largeembedding_2048.MoCo(
        model,
        teacher_model,
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args)
    print(model)
    #for name, param in model.named_parameters():
        #print (name, param.requires_grad)
    #    param.requires_grad = True

    #print ("testing grad requires")
    #params_list = list(model.named_parameters())
    #print (len(params_list))
    #print (params_list[0][1].requires_grad)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_mse = nn.MSELoss(reduction='sum').cuda(args.gpu)

    #print ("----model parameters------")
    #print (model.parameters())
    #print ("--------------------------")

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint_path = get_last_checkpoint(args.resume)
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(checkpoint_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(checkpoint_path, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            out = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(out)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Resume from ./ckpt
    if args.resume_ckpt:
        checkpoint_path = get_last_checkpoint('./ckpt', args)
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(checkpoint_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(checkpoint_path, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            out = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(out)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.teacher:
        if os.path.isfile(args.teacher):
            print("=> loading teacher checkpoint '{}'".format(args.teacher))
            if args.gpu is None:
                teacher_checkpoint = torch.load(args.teacher)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                teacher_checkpoint = torch.load(args.teacher, map_location=loc)

            if args.teacher_arch in ['resnet50w2', 'SWAVresnet50',
                                     'DCresnet50', 'SELAresnet50']:
                param_dict = teacher_checkpoint
            else:
                param_dict = teacher_checkpoint['state_dict']

            new_param_dict = {}

            for k,v in param_dict.items():
                #print (k)
                if args.teacher_arch in ['resnet50w2', 'SWAVresnet50',
                                         'DCresnet50', 'SELAresnet50']:
                    kq = k.replace("module.","module.teacher_encoder_q.")
                    kk = k.replace("module.","module.teacher_encoder_k.")
                    new_param_dict[kk] = v
                    new_param_dict[kq] = v
                else:
                    k = k.replace("encoder","teacher_encoder")
                    new_param_dict[k] = v
                #if "encoder_k" in k:
                #    k = k.replace("fc.2", "fc1")

            model.load_state_dict(new_param_dict, strict=False)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    train_dataset = datasets.ImageFolder(
        traindir,
        moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    scaler = GradScaler()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, criterion_mse, optimizer, epoch, args, scaler)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='./ckpt/checkpoint_{}_{}_{:04d}.pth.tar'.format(args.arch, args.teacher_arch, epoch), pre_filename='./ckpt/checkpoint_{}_{}_{:04d}.pth.tar'.format(args.arch, args.teacher_arch, epoch-1), args=args)


def train(train_loader, model, criterion, criterion_mse, optimizer, epoch, args, scaler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mse_losses = AverageMeter('MSE Loss', ':.4e')
    mse_losses_qk = AverageMeter('MSE Loss KQ', ':.4e')
    mse_losses_q1 = AverageMeter('MSE Loss Q1', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, mse_losses, mse_losses_qk, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        with autocast(enabled=True):
            output, target, student_q, teacher_q, student_qk, teacher_qk = model(im_q=images[0], im_k=images[1])
            loss = criterion(output, target)
            mse_loss = criterion_mse(teacher_q, student_q)
            mse_loss_qk = criterion_mse(teacher_qk, student_qk)

            if args.only_mse:
                loss = 0.0 * loss + 1.0 * mse_loss + 1.0 * mse_loss_qk
            else:
                loss = 1.0 * loss + 1.0 * mse_loss + 1.0 * mse_loss_qk

        #loss += mse_loss
        #loss += mse_loss_qk
        #loss += mse_loss_q1

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), images[0].size(0))
        mse_losses.update(mse_loss.item(), images[0].size(0))
        mse_losses_qk.update(mse_loss_qk.item(), images[0].size(0))
        #mse_losses_q1.update(mse_loss_q1.item(), images[0].size(0))

        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        if args.arch == 'mobilenetv3':
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', pre_filename=None, args=None):
    torch.save(state, filename)
    if os.path.exists(pre_filename):
        os.remove(pre_filename)
    if is_best:
        shutil.copyfile(filename, f'model_best_{args.arch}_{args.teacher_arch}.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_last_checkpoint(checkpoint_dir, args=None):
    all_ckpt = glob.glob(os.path.join(checkpoint_dir, f'checkpoint_{args.arch}_{args.teacher_arch}_0*.pth.tar'))
    if all_ckpt:
        all_ckpt = sorted(all_ckpt)
        return all_ckpt[-1]
    else:
        return ''

if __name__ == '__main__':
    main()
