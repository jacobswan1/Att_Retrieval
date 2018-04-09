'''Train Sun Attribute with PyTorch.'''
from __future__ import print_function

import torch
import torch.optim as optim
import argparse


def parse_opts():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--msg', default=False, type=bool, help='display message')
    parser.add_argument('--resume', '-r',  default=True, type=bool, help='resume from checkpoint')
    parser.add_argument('--use_gpu', default=torch.cuda.is_available(), type=bool, help='Use GPU or not')
    parser.add_argument('--multi_gpu', default=1, type=bool, help='Use multi-GPU or not')
    parser.add_argument('--gpu_id', default=-1, type=int, help='Use specific GPU.')

    parser.add_argument('--optimizer', default=optim.SGD, help='optimizer')
    parser.add_argument('--num_workers', default=2, type=int, help='num of fetching threads')
    parser.add_argument('--batch_size', default=48, type=int, help='batch size')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='weight decay')
    parser.add_argument('--seed', default=0, type=int, help='randome seed')
    parser.add_argument('--result_path', default='./results', help='result path')
    parser.add_argument('--checkpoint_path', default='./checkpoint', help='checkpoint path')
    parser.add_argument('--pretrain', default='best_classifi_95.57.pth', help='result path')

    parser.add_argument('--matplotlib_mode',default='Agg',help='matplotlib mode (TkAgg|Agg)')#

    parser.add_argument('--checkpoint_epoch', default=50, type=int, help='epochs to save checkpoint ')
    parser.add_argument('--lr_adjust_epoch', default=100, type=int, help='lr adjust epoch')
    parser.add_argument('--lr_weight', default=1e-4, type=int, help='default weight of the graph loss')
    parser.add_argument('--n_epoch', default=1000, type=int, help='training epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

    parser.add_argument('--network', default='preactres', help='dataset(res152|preactres)')

    args = parser.parse_args()

    return args
