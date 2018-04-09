from __future__ import print_function

import os
import time
import csv
import torch
import torch.nn as nn
from torch.autograd import Variable




def set_parameters(opts):
    '''
    This function is called before training/testing to set parameters
    :param opts:
    :return opts:
    '''

    if not opts.__contains__('train_losses'):
        opts.train_losses=[]

    if not opts.__contains__('train_accuracies'):
        opts.train_accuracies = []

    if not opts.__contains__('valid_losses'):
        opts.valid_losses = []
    if not opts.__contains__('valid_accuracies'):
        opts.valid_accuracies = []

    if not opts.__contains__('test_losses'):
        opts.test_losses = []
    if not opts.__contains__('test_accuracies'):
        opts.test_accuracies = []

    if not opts.__contains__('best_acc'):
        opts.best_acc = 0.0

    if not opts.__contains__('lowest_loss'):
        opts.lowest_loss = 1e4

    if not opts.__contains__('checkpoint_path'):
        opts.checkpoint_path = 'checkpoint'

    if not os.path.exists(opts.checkpoint_path):
        os.mkdir(opts.checkpoint_path)

    if not opts.__contains__('checkpoint_epoch'):
        opts.checkpoint_epoch = 5

    if not opts.__contains__('valid_pearson_r'):
        opts.valid_pearson_r = []

    if not opts.__contains__('test_pearson_r'):
        opts.test_pearson_r = []


def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y

    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y.float())
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def train_net(net, opts):

    print('training at epoch {}'.format(opts.epoch+1))

    if opts.use_gpu:
        net.cuda()
        if opts.multi_gpu:
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    net.train()
    train_loss = 0
    total_time = 0
    data_time = 0
    total = 0
    correct = 0
    extra = 0.
    
    optimizer = opts.current_optimizer

    end_time = time.time()

    for batch_idx, (inputs, targets, attribute) in enumerate(opts.data_loader):
        if opts.use_gpu:
            inputs, targets, attribute = inputs.cuda(async=True), targets.cuda(async=True), attribute.cuda(async=True)

        # loading time
        data_time += (time.time() - end_time)
        optimizer.zero_grad()  # flush

        # ff
        inputs = Variable(inputs)
        targets = Variable(targets).long()
        attribute = Variable(attribute).float()
        outputs, feat = net(inputs)
        # loss = opts.criterion[0](feat, attribute)
        loss = opts.criterion[0](outputs, targets) + opts.criterion[1](feat, attribute)
        if batch_idx % 100 == 0:
            print('graph_loss: %.8f ' % opts.criterion[1](feat, attribute).data[0])

        train_loss += loss.data[0]

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # Bp
        loss.backward()
        optimizer.step()

        total_time += (time.time() - end_time)
        end_time = time.time()

        if opts.msg:
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        opts.train_batch_logger.log({
            'epoch': (opts.epoch+1),
            'batch': batch_idx + 1,
            'loss': train_loss / (batch_idx + 1),
            'acc': correct / total,
            'extra': extra/(batch_idx + 1)
        })

    train_loss /= (batch_idx + 1)

    opts.train_epoch_logger.log({
        'epoch': (opts.epoch+1),
        'loss': train_loss,
        'acc': correct / total,
        'time': total_time,
        'extra': extra / (batch_idx + 1)
    })

    print('Loss: %.3f | Acc: %.3f%% (%d/%d), elasped time: %3.f seconds.'
          % (train_loss, 100. * correct / total, correct, total, total_time))
    opts.train_accuracies.append(correct / total)

    opts.train_losses.append(train_loss)


def eval_net(net, opts):
    if opts.validating:
        print('Validating at epoch {}'.format(opts.epoch + 1))

    if opts.testing:
        print('Testing at epoch {}'.format(opts.epoch + 1))

    if opts.use_gpu:
        net.cuda()
        if opts.multi_gpu:
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    if not opts.__contains__('validating'):
        opts.validating = False
    if not opts.__contains__('testing'):
        opts.testing = False

    net.eval()
    eval_loss = 0
    correct = 0
    total = 0
    total_time = 0
    extra = 0.

    end_time = time.time()

    for batch_idx, (inputs, targets, attribute) in enumerate(opts.data_loader):
        if opts.use_gpu:
            inputs, targets, attribute = inputs.cuda(async=True), targets.cuda(async=True), attribute.cuda(async=True)

        inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True).long()
        attribute = Variable(attribute).float()
        outputs, feat = net(inputs)

        # cross-entropy loss
        loss = opts.criterion[0](outputs, targets) + opts.criterion[1](feat, attribute)
        # loss = opts.criterion[0](feat, attribute)

        eval_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        total_time += (time.time() - end_time)
        end_time = time.time()

        if opts.msg:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (eval_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    eval_loss /= (batch_idx + 1)
    eval_acc = correct / total

    if opts.testing:
        opts.test_losses.append(eval_loss)
        opts.test_accuracies.append(correct / total)

        opts.test_epoch_logger.log({
            'epoch': (opts.epoch + 1),
            'loss': eval_loss,
            'acc': correct / total,
            'time': total_time,
            'extra': extra/(batch_idx + 1)
        })

    if opts.validating:
        opts.valid_losses.append(eval_loss)
        opts.valid_accuracies.append(correct / total)

        opts.valid_epoch_logger.log({
            'epoch': (opts.epoch + 1),
            'loss': eval_loss,
            'acc': correct / total,
            'time': total_time
        })
    # Save checkpoint.

    states = {
        'state_dict': net.state_dict(),
        'epoch': opts.epoch+1,
        'train_losses': opts.train_losses,
        'optimizer': opts.current_optimizer.state_dict()
    }

    if opts.__contains__('acc'):
        states['acc'] = eval_acc,

    if opts.__contains__('valid_losses'):
        states['valid_losses']=opts.valid_losses
    if opts.__contains__('test_losses'):
        states['test_losses'] = opts.test_losses

    if eval_acc > opts.best_acc:
        if not os.path.isdir(opts.checkpoint_path):
            os.mkdir(opts.checkpoint_path)
        torch.save(states, os.path.join(opts.checkpoint_path, 'best_net.pth'))
        opts.best_acc = eval_acc

    if opts.epoch % opts.checkpoint_epoch == 0:
        save_file_path = os.path.join(opts.checkpoint_path, '_{}.pth'.format(opts.epoch))
        torch.save(states, save_file_path)

    print('Loss: %.3f | Acc: %.3f%% (%d/%d), elasped time: %3.f seconds. Best Acc: %.3f%%'
          % (eval_loss, 100. * correct / total, correct, total, total_time, opts.best_acc*100))


class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()
