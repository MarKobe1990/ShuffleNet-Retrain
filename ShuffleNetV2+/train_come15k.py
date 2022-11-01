import os
import sys
import torch
import torch.nn as nn
import numpy as np
import PIL
from PIL import Image
import time
import logging
import warnings
import argparse

from torchvision.transforms import transforms

from network import ShuffleNetV2_Plus
from COME15KClassDataset import set_data_loader
from tqdm import tqdm
from utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters

warnings.filterwarnings("ignore")


class ToBGRTensor(object):

    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:, :, ::-1]  # 2 BGR
        img = np.transpose(img, [2, 0, 1])  # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img


class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]


def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def train(model, device, args, epoch, bn_process=False, all_iters=None, total_iters=None):
    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_loader = args.train_loader

    Top1_err = 0.0
    model.train()
    pbar = tqdm(train_loader, desc='Epoch-train' + str(epoch), unit='batch')

    for iters, (batched_inputs_img, batched_inputs_label) in enumerate(pbar):
        pbar.set_description('Epoch-train:' + str(epoch))
        scheduler.step()
        if bn_process:
            adjust_bn_momentum(model, iters)
        all_iters += 1
        d_st = time.time()

        batched_inputs_label = batched_inputs_label.type(torch.LongTensor)
        batched_inputs_img, batched_inputs_label = batched_inputs_img.to(device), batched_inputs_label.to(device)
        data_time = time.time() - d_st

        output = model(batched_inputs_img)
        loss = loss_function(output, batched_inputs_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prec1 = accuracy(output, batched_inputs_label)

        Top1_err += 1 - prec1.item() / 100

        if all_iters % args.display_interval == 0:
            printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters, scheduler.get_lr()[0],
                                                                               loss.item()) + \
                        'Top-1 err = {:.6f},\t'.format(Top1_err / args.display_interval) + \
                        'data_time = {:.6f},\ttrain_time = {:.6f}'.format(data_time,
                                                                          (time.time() - t1) / args.display_interval)
            logging.info(printInfo)
            t1 = time.time()
            Top1_err, Top5_err = 0.0, 0.0

        save_checkpoint({
                'state_dict': model.state_dict(),
            }, epoch)

    return all_iters


def validate(model, device, args, epoch):
    objs_easy = AvgrageMeter()
    top1_easy = AvgrageMeter()

    objs_hard = AvgrageMeter()
    top1_hard = AvgrageMeter()


    loss_function = args.loss_function
    val_loader_easy = args.val_loader_easy
    val_loader_hard = args.val_loader_hard
    pbar_easy = tqdm(val_loader_easy, desc='Epoch-val-easy' + str(epoch), unit='img')
    pbar_hard = tqdm(val_loader_hard, desc='Epoch-val-hard' + str(epoch), unit='img')
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for idx, (data, target) in enumerate(pbar_easy):
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            prec1 = accuracy(output, target, topk=(1, ))
            n = data.size(0)
            objs_easy.update(loss.item(), n)
            top1_easy.update(prec1.item(), n)


        logInfo = 'val-easy Epoch {}: loss = {:.6f},\t'.format(epoch, objs_easy.avg) + \
              'Top-1 err = {:.6f},\t'.format(1 - top1_easy.avg / 100) + \
              'val_time = {:.6f}'.format(time.time() - t1)
        logging.info(logInfo)

        for idx, (data, target) in enumerate(pbar_hard):
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            prec1 = accuracy(output, target, topk=(1,))
            n = data.size(0)
            objs_hard.update(loss.item(), n)
            top1_hard.update(prec1.item(), n)

        logInfo = 'val-easy Epoch {}: loss = {:.6f},\t'.format(epoch, objs_hard.avg) + \
                  'Top-1 err = {:.6f},\t'.format(1 - top1_hard.avg / 100) + \
                  'val_time = {:.6f}'.format(time.time() - t1)
        logging.info(logInfo)



def load_checkpoint(net, checkpoint):
    from collections import OrderedDict

    temp = OrderedDict()
    if 'state_dict' in checkpoint:
        checkpoint = dict(checkpoint['state_dict'])
    for k in checkpoint:
        k2 = 'module.' + k if not k.startswith('module.') else k
        temp[k2] = checkpoint[k]

    net.load_state_dict(temp, strict=True)


def main():
    args = get_args()

    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(
        os.path.join('log/train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True

    # data_loader
    assert os.path.exists(args.train_dir)
    train_transforms_compose = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # RGB,imageNet1k mean and standard
    ])
    args.train_loader = set_data_loader(dataset_attr_word="train", batch_size=10, size=256, shuffle=True,
                                        transforms_compose=train_transforms_compose)
    assert os.path.exists(args.val_dir)
    val_transforms_compose = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # RGB,imageNet1k mean and standard
    ])

    args.val_loader_easy = set_data_loader(dataset_attr_word="val_easy", batch_size=1, size=256, shuffle=False,
                                      transforms_compose=val_transforms_compose)
    args.val_loader_hard = set_data_loader(dataset_attr_word="val_hard", batch_size=1, size=256, shuffle=False,
                                           transforms_compose=val_transforms_compose)
    print('load data successfully')

    # init model
    architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    model = ShuffleNetV2_Plus(architecture=architecture, model_size=args.model_size)
    pre_train_model_weight_dic = {
        'Small': 'shuffle_net_v2_plus_image1K_pretrianed_weight/ShuffleNetV2+.ImageNet1k_pre_trained_Small.pth.tar',
        'Medium': 'shuffle_net_v2_plus_image1K_pretrianed_weight/ShuffleNetV2+.ImageNet1k_pre_trained_Medium.pth.tar',
        'Large': 'shuffle_net_v2_plus_image1K_pretrianed_weight/ShuffleNetV2+.ImageNet1k_pre_trained_Large.pth.tar'
    }

    if args.fine_tune:
        pre_train_weight = torch.load(pre_train_model_weight_dic[args.model_size])
        model.load_state_dict(pre_train_weight)

    # 更新器
    optimizer = torch.optim.SGD(get_parameters(model),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # criterion_smooth = CrossEntropyLabelSmooth(8, 0.1)
    criterion = nn.CrossEntropyLoss()
    if use_gpu:
        loss_function = criterion.cuda()
        device = torch.device("cuda")
    else:
        loss_function = criterion
        device = torch.device("cpu")

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda step: (
                                                          1.0 - step / args.total_iters) if step <= args.total_iters else 0,
                                                  last_epoch=-1)

    model = model.to(device)

    all_iters = 0
    if args.auto_continue:
        lastest_model, iters = get_lastest_model()
        if lastest_model is not None:
            all_iters = iters
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint')
            for i in range(iters):
                scheduler.step()

    args.optimizer = optimizer
    args.loss_function = loss_function
    args.scheduler = scheduler
    total_iters = args.total_epoch * args.train_loader.__len__() / args.batch_size
    for epoch in range(1, args.total_epoch + 1):
        all_iters = train(model, device, args, epoch, bn_process=True, all_iters=all_iters, total_iters=total_iters)

        validate(model, device, args, epoch, all_iters=all_iters)

        save_checkpoint({'state_dict': model.state_dict(), }, args.total_epoch, tag='bnps-')


def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV2_Plus")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-resume', type=str, default='./snet_detnas.pkl', help='path for eval model')
    parser.add_argument('--batch-size', type=int, default=10, help='batch size')
    parser.add_argument('--total-epoch', type=int, default=100, help='total iters')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')

    parser.add_argument('--auto-continue', type=bool, default=True, help='auto continue')
    parser.add_argument('--display-interval', type=int, default=200, help='display interval')
    # parser.add_argument('--val-interval', type=int, default=800, help='val interval')
    # parser.add_argument('--save-interval', type=int, default=800, help='save interval')

    parser.add_argument('--model-size', type=str, default='Large', choices=['Small', 'Medium', 'Large'],
                        help='size of the model')
    parser.add_argument('--train-dir', type=str, default='data/SOD-SemanticDataset/train',
                        help='path to training dataset')
    parser.add_argument('--val-dir', type=str, default='data/SOD-SemanticDataset/train/val',
                        help='path to validation dataset')
    parser.add_argument('--fine-tune', type=bool, default=True, help='load pretrain weight at start')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
