# coding: utf-8

import argparse
import os
import time
import math
import json
import shutil

from data.icdar import collate_fn, ICDAR
from data.augment import east_aug
from loss import LossFunc
from utils import save_checkpoint, AverageMeter, init_weights, get_images
from models import East, East_Resnet18
import config as args
from inference import predict
from pyicdartools import evaluation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter


best_loss = float('inf')
best_f1 = 0.0
if args.tensorboardX:
    writer = SummaryWriter(args.tensorboardX)


def set_learning_rate(optimizer, epoch, iter_size, iter_index, args):
    current_iter = epoch * iter_size + iter_index
    if current_iter < args.warm_up:
        current_lr = args.lr * math.pow(current_iter / args.warm_up, 4)
    else:
        current_lr = args.lr * (1 + math.cos(epoch * math.pi / args.epochs)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return current_lr


def train(dataloader, model, criterion, optimizer, epoch, args):
    model.train()

    losses = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    tic = time.time()
    for i, (img, score_map, geo_map, training_mask) in enumerate(dataloader):
        data_time.update(time.time() - tic)

        lr = set_learning_rate(optimizer, epoch, len(dataloader), i, args)
        img = img.cuda(non_blocking=True)
        score_map = score_map.cuda()
        geo_map = geo_map.cuda()
        training_mask = training_mask.cuda()

        f_score, f_geometry = model(img)
        loss = criterion(score_map, f_score, geo_map, f_geometry, training_mask)
        losses.update(loss.item(), img.size(0))

        if args.tensorboardX:
            cur_iter = epoch * len(dataloader) + i
            writer.add_scalar('Train/Loss', loss.item(), cur_iter)
            writer.add_scalar('Train/Lr', lr, cur_iter)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - tic)
        tic = time.time()

        if i % args.print_freq == 0:
            print(time.strftime('%m/%d %H:%M:%S', time.localtime()), end='\t')
            print('Train Epoch [{0}][{1}/{2}] '
                  'Batch Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                  'Data Time {data_time.val:.3f}({data_time.avg:.3f}) '
                  'Loss {loss.val:.3f}({loss.avg:.3f}) '
                  'Lr {lr:.6f}'.format(
                  epoch, i, len(dataloader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  lr=lr), flush=True)


def val(dataloader, model, criterion, args):
    print('val...')
    model.eval()

    losses = AverageMeter()
    batch_time = AverageMeter()

    tic = time.time()
    with torch.no_grad():
        for i, (img, score_map, geo_map, training_mask) in enumerate(dataloader):
            img = img.cuda(non_blocking=True)
            score_map = score_map.cuda()
            geo_map = geo_map.cuda()
            training_mask = training_mask.cuda()

            f_score, f_geometry = model(img)
            loss = criterion(score_map, f_score, geo_map, f_geometry, training_mask)
            losses.update(loss.item(), img.size(0))

            batch_time.update(time.time() - tic)
            tic = time.time()

            if i % args.print_freq == 0:
                print(time.strftime('%m/%d %H:%M:%S', time.localtime()), end='\t')
                print('Eval [{0}/{1}] '
                      'Batch Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                      'Loss {loss.val:.3f}({loss.avg:.3f})'.format(
                      i, len(dataloader),
                      batch_time=batch_time,
                      loss=losses), flush=True)

    return losses.avg


def val_f1(model, args):
    if args.val_dir:
        image_list = get_images(args.val_dir)
    else:
        image_list = []
        with open(args.val_annotation, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                image_path = line['url']
                if (image_path.startswith('qiniu:///')):
                    image_path = image_path.replace('qiniu:///', '/workspace/mnt/bucket/', 1)
                image_list.append(image_path)

    predict(model, image_list, './tmp/tmp_res.json',
            input_size=args.inference_size,
            thresholds=(0.8, 0.1, 0.2),
            draw_dir='./tmp/draw')

    metrics = evaluation.eval(args.val_annotation, './tmp/tmp_res.json')

    return metrics['method']['hmean'], metrics['method']['precision'], metrics['method']['recall']


def main(args):
    # cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    CUDA_COUNT = torch.cuda.device_count()
    print('Using {} GPU(s)...'.format(CUDA_COUNT))

    # trainset
    trainset = ICDAR(args.train_dir, args.train_annotation,
                     input_size=args.input_size,
                     min_text_size=args.min_text_size,
                     transforms=east_aug)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=args.num_workers)

    if args.val_annotation:
        os.makedirs('./tmp/draw', exist_ok=True)
        # valset
        valset = ICDAR(args.val_dir, args.val_annotation,
                       input_size=args.input_size, min_text_size=args.min_text_size)
        valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=args.num_workers)

    if args.backbone == 'resnet18':
        net = East_Resnet18
    else:
        net = East
    model = net(pretrained_model=args.backbone_pretrain, text_scale=args.text_scale)
    model = nn.DataParallel(model).cuda()
    criterion = LossFunc().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)

    if args.checkpoint:
        # resume
        print("=> loading checkpoint from {}...".format(args.checkpoint))
        state = torch.load(args.checkpoint)
        model.load_state_dict(state['model_state_dict'])
        if not args.finetune:
            args.start_epoch = state['epoch']
            optimizer.load_state_dict(state['optimizer'])
        else:
            args.start_epoch = 0
        print('=> start epoch: {}'.format(args.start_epoch))
    else:
        args.start_epoch = 0
        # init weight
        init_weights(model, init_type='xavier')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    for epoch in range(args.start_epoch, args.epochs):
        global best_loss
        global best_f1
        train(trainloader, model, criterion, optimizer, epoch, args)
        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        # checkpoint_file = os.path.join(args.checkpoint_dir,
        #                                'checkpoint_epoch_{:04d}.pth.tar'.format(state['epoch']))
        checkpoint_file = os.path.join(args.checkpoint_dir, 'checkpoint.pth.tar')
        torch.save(state, checkpoint_file)

        if args.val_annotation and (epoch + 1) % args.eval_interval == 0:
            # validate
            # val_loss = val(valloader, model, criterion, args)
            # is_best = (val_loss < best_loss)
            f1, precision, recall = val_f1(model, args)

            if args.tensorboardX:
                writer.add_scalar('Val/F1', f1, epoch)
                writer.add_scalar('Val/precision', precision, epoch)
                writer.add_scalar('Val/recall', recall, epoch)

            is_best = (f1 > best_f1)
            if is_best:
                # best_loss = val_loss
                best_f1 = f1
                print('saving...')
                best_checkpoint_file = os.path.join(args.checkpoint_dir,
                                                    'checkpoint_best.pth.tar')
                shutil.copy2(checkpoint_file, best_checkpoint_file)


def print_args(args):
    print('\nArgs:')
    print('-'*80)
    for k in dir(args):
        if not k.startswith('_'):
            print('{} : {}'.format(k, args.__dict__[k]))
    print('-'*80)
    print('\n')


def parse():
    parser = argparse.ArgumentParser('EAST')
    parser.add_argument('--train_annotation', type=str)
    parser.add_argument('--val_annotation', type=str)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet18'])
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--input_size', type=int)
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument('--checkpoint', type=str)
    return parser.parse_args()


def update_args(old_args, new_args):
    for arg_k in dir(new_args):
        if not arg_k.startswith('_') and new_args.__dict__[arg_k] is not None:
            old_args.__dict__[arg_k] = new_args.__dict__[arg_k]


if __name__ == "__main__":
    new_args = parse()
    update_args(args, new_args)
    print_args(args)
    main(args)
