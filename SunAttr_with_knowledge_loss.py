'''Train Sun Attribute with PyTorch.'''
from __future__ import print_function
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from parser import *
from net_util import *
from resnet import resnet101
from graph_loss import GraphLoss
from collections import OrderedDict
from torchvision import transforms
from SunAttributeDataset import SunAttributeDataset

if __name__ == '__main__':

    opts = parse_opts()

    if opts.gpu_id >= 0:
        torch.cuda.set_device(opts.gpu_id)
        opts.multi_gpu = False

    matplotlib.use(opts.matplotlib_mode)

    torch.manual_seed(opts.seed)
    if opts.use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed(opts.seed)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    print("| Preparing SUN Attribute dataset...")
    img_path = '/media/drive1/Data/Sun_attri/SUNAttributeDB/images.mat'
    attri_path = '/media/drive1/Data/Sun_attri/SUNAttributeDB/attributeLabels_continuous.mat'
    root_dir = '/media/drive1/Data/Sun_attri/images/'
    trainset = SunAttributeDataset(img_path, attri_path, root_dir, transform_train)
    opts.num_classes = 611

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    test_loader = torch.utils.data.DataLoader(trainset, batch_size=opts.batch_size, shuffle=True,
                                              num_workers=opts.num_workers)

    if not os.path.exists(opts.result_path):
        os.mkdir(opts.result_path)

    opts.train_epoch_logger = Logger(os.path.join(opts.result_path, 'train.log'), ['epoch', 'time', 'loss', 'acc', 'extra'])
    opts.train_batch_logger = Logger(os.path.join(opts.result_path, 'train_batch.log'), ['epoch', 'batch', 'loss', 'acc', 'extra'])
    opts.test_epoch_logger = Logger(os.path.join(opts.result_path, 'test.log'), ['epoch', 'time', 'loss', 'acc', 'extra'])

    # Model
    print('==> Building model...')
    net = resnet101(pretrained=True)
    net.fc = nn.Linear(2048, opts.num_classes)
    if opts.pretrain:
        # load pretrained cifar model
        path = os.path.join(os.getcwd(), 'checkpoint', opts.pretrain)
        tnet = torch.load(path)
        new_state_dict = OrderedDict()
        for value in tnet['state_dict']:
            key = value.replace("module.", "")
            value = tnet['state_dict'][value]
            new_state_dict[key] = value
        net.load_state_dict(new_state_dict)

    start_epoch = 0
    print('==> model built.')
    net.cuda()
    gl = GraphLoss(opts.lr_weight)

    # opts.criterion = [gl]
    opts.criterion = [nn.CrossEntropyLoss(), gl]

    # Training
    print(opts)
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in parameters])
    print(params, 'trainable parameters in the network.')
    set_parameters(opts)

    for epoch in range(start_epoch, start_epoch+opts.n_epoch):
        opts.epoch = epoch
        if epoch == 0:
            opts.data_loader = train_loader
            parameters = filter(lambda p: p.requires_grad, net.parameters())
            if opts.optimizer == optim.SGD:
                opts.current_optimizer = opts.optimizer(parameters, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
            else:
                opts.current_optimizer = opts.optimizer(parameters, lr=opts.lr, weight_decay=opts.weight_decay)

        if (epoch % 150 == 0) and (epoch is not 0):
            parameters = filter(lambda p: p.requires_grad, net.parameters())
            opts.lr /= 10
            opts.current_optimizer = opts.optimizer(parameters, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

        opts.data_loader = train_loader
        train_net(net, opts)
        opts.data_loader = train_loader
        opts.validating = False
        opts.testing = True
        eval_net(net, opts)

        if (epoch + 1) % opts.checkpoint_epoch == 0:
            plt.ion()
            plt.figure(1)
            plt.subplot(1, 2, 1)
            plt.title('Loss Plot', fontsize=10)
            plt.xlabel('Epochs', fontsize=10)
            plt.ylabel('Loss', fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.plot(opts.train_losses, 'b')
            if opts.__contains__('test_losses'):
                plt.plot(opts.test_losses, 'r')
            if opts.__contains__('valid_losses'):
                plt.plot(opts.valid_losses, 'g')
            plt.subplot(1, 2, 2)
            plt.title('Accuracy Plot', fontsize=10)
            plt.xlabel('Epochs', fontsize=10)
            plt.ylabel('Accuracy', fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.plot(opts.train_accuracies, 'b')
            if opts.__contains__('test_accuracies'):
                plt.plot(opts.test_accuracies, 'r')
            if opts.__contains__('valid_accuracies'):
                plt.plot(opts.valid_accuracies, 'g')
            plt.savefig(os.path.join(opts.result_path, 'TrainingPlots'))
            plt.show()
