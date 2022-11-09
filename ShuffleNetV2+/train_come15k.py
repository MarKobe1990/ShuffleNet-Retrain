import datetime
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import PIL
from PIL import Image
import time
import logging
import warnings
import argparse

from tensorboardX import SummaryWriter
from torchvision.transforms import transforms

from network import ShuffleNetV2_Plus
from COME15KClassDataset import set_data_loader
from tqdm import tqdm
from utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters

warnings.filterwarnings("ignore")


def init_tb_writer_global(args):
    tb_log_save_dir = args.save + '/tb_log_global/'
    writer_global = SummaryWriter(tb_log_save_dir, comment="writer_global",
                                  filename_suffix="_writer_global")
    return writer_global


def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def train(model, device, args, epoch, writer, bn_process=False, all_iters=None, total_iters=None):
    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_loader = args.train_loader

    Top1_err = 0.0
    Top2_err = 0.0
    Top3_err = 0.0
    model.train()
    pbar = tqdm(train_loader, desc='Epoch-train' + str(epoch), unit='batch')
    d_st = time.time()
    for iters, (batched_inputs_img, batched_inputs_label) in enumerate(pbar):
        if bn_process:
            adjust_bn_momentum(model, iters + 1)
        all_iters += 1

        batched_inputs_label = batched_inputs_label.type(torch.LongTensor)
        batched_inputs_img, batched_inputs_label = batched_inputs_img.to(device), batched_inputs_label.to(device)

        output = model(batched_inputs_img)
        loss = loss_function(output, batched_inputs_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prec1, prec2, prec3 = accuracy(output, batched_inputs_label, topk=(1, 2, 3))
        pbar.set_postfix({
            "iter": str(all_iters) + "/" + str(total_iters),
            "lr": scheduler.get_lr()[0],
            "loss": loss.item(),
            "prec1": prec1.item() / 100,
            "prec2": prec2.item() / 100,
            "prec3": prec3.item() / 100

        })

        Top1_err += 1 - prec1.item() / 100
        Top2_err += 1 - prec2.item() / 100
        Top3_err += 1 - prec3.item() / 100

    scheduler.step()
    train_time = time.time() - d_st

    printInfo = 'TRAIN epoch {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_lr()[0],
                                                                        loss.item()) + \
                'Top-1 err = {:.6f},\t'.format(Top1_err / iters) + \
                'Top-2 err = {:.6f},\t'.format(Top2_err / iters) + \
                'Top-3 err = {:.6f},\t'.format(Top3_err / iters) + \
                'data_time = {:.6f},\ttrain_time = {:.6f}\n'.format(time.time(),
                                                                    train_time / iters)
    logging.info(printInfo)
    print(printInfo)

    hyer_params_dic = {
        "lr": scheduler.get_lr()[0],
        "epoch": epoch
    }
    result_dic = {
        "loss": loss.item(),
        "top-1-error": Top1_err / iters,
        "top-2-error": Top2_err / iters,
        "top-3-error": Top3_err / iters
    }
    writer.add_hparams(hyer_params_dic, result_dic, name='train', global_step=epoch)
    writer.add_scalars(main_tag='train', tag_scalar_dict=result_dic, global_step=epoch)

    save_checkpoint(args.save, {
        'state_dict': model.state_dict(),
    }, args.total_epoch, tag='current-')

    return all_iters


def validate_easy(model, device, args, epoch, writer):
    objs_easy = AvgrageMeter()
    top1_easy = AvgrageMeter()
    top2_easy = AvgrageMeter()
    top3_easy = AvgrageMeter()
    loss_function = args.loss_function
    val_loader_easy = args.val_loader_easy

    pbar_easy = tqdm(val_loader_easy, desc='Epoch-val-easy:epoch-' + str(epoch), unit='img')
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for idx, (data, target) in enumerate(pbar_easy):
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            prec1, prec2, prec3 = accuracy(output, target, topk=(1, 2, 3))
            n = data.size(0)
            objs_easy.update(loss.item(), n)
            top1_easy.update(prec1.item(), n)
            top2_easy.update(prec2.item(), n)
            top3_easy.update(prec3.item(), n)
            pbar_easy.set_postfix(
                {
                    'process': str(idx) + "/" + str(len(pbar_easy)),
                    'prec1': prec1.item(),
                    'prec2': prec2.item(),
                    'prec3': prec3.item()
                })

        logInfo = 'val-easy Epoch {}: loss = {:.6f},\t'.format(epoch, objs_easy.avg) + \
                  'Top-1 err = {:.6f},\t'.format(1 - top1_easy.avg / 100) + \
                  'Top-2 err = {:.6f},\t'.format(1 - top2_easy.avg / 100) + \
                  'Top-3 err = {:.6f},\t'.format(1 - top3_easy.avg / 100) + \
                  'val_time = {:.6f}\n'.format(time.time() - t1)
        print(logInfo)
        logging.info(logInfo)

        hyer_params_dic = {
            "epoch": epoch
        }
        result_dic = {
            "loss": loss.item(),
            "top-1-error": (1 - top1_easy.avg / 100),
            "top-2-error": (1 - top2_easy.avg / 100),
            "top-3-error": (1 - top3_easy.avg / 100)
        }
        writer.add_hparams(hyer_params_dic, result_dic, name='test_easy', global_step=epoch)
        writer.add_scalars(main_tag='test/easy', tag_scalar_dict=result_dic, global_step=epoch)


def validate_hard(model, device, args, epoch, writer):
    objs_hard = AvgrageMeter()
    top1_hard = AvgrageMeter()
    top2_hard = AvgrageMeter()
    top3_hard = AvgrageMeter()

    loss_function = args.loss_function

    val_loader_hard = args.val_loader_hard

    pbar_hard = tqdm(val_loader_hard, desc='Epoch-val-hard:epoch-' + str(epoch), unit='img')
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for idx, (data, target) in enumerate(pbar_hard):
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            prec1, prec2, prec3 = accuracy(output, target, topk=(1, 2, 3))
            n = data.size(0)
            objs_hard.update(loss.item(), n)
            top1_hard.update(prec1.item(), n)
            top2_hard.update(prec2.item(), n)
            top3_hard.update(prec3.item(), n)
            pbar_hard.set_postfix({
                'process': str(idx) + "/" + str(len(pbar_hard)),
                'prec1': prec1.item(),
                'prec2': prec2.item(),
                'prec3': prec3.item()
            }, refresh=True)
        logInfo = 'val-hard Epoch {}: loss = {:.6f},\t'.format(epoch, objs_hard.avg) + \
                  'Top-1 err = {:.6f},\t'.format(1 - top1_hard.avg / 100) + \
                  'Top-2 err = {:.6f},\t'.format(1 - top2_hard.avg / 100) + \
                  'Top-3 err = {:.6f},\t'.format(1 - top3_hard.avg / 100) + \
                  'val_time = {:.6f}\n'.format(time.time() - t1)
        print(logInfo)
        logging.info(logInfo)
        hyer_params_dic = {
            "epoch": epoch
        }
        result_dic = {
            "loss": loss.item(),
            "top-1-error": (1 - top1_hard.avg / 100),
            "top-2-error": (1 - top2_hard.avg / 100),
            "top-3-error": (1 - top3_hard.avg / 100)
        }
        writer.add_hparams(hyer_params_dic, result_dic, name='test_hard', global_step=epoch)
        writer.add_scalars(main_tag='test/hard', tag_scalar_dict=result_dic, global_step=epoch)


def load_checkpoint(net, checkpoint):
    from collections import OrderedDict

    temp = OrderedDict()
    if 'state_dict' in checkpoint:
        checkpoint = dict(checkpoint['state_dict'])
    for k in checkpoint:
        k2 = 'module.' + k if not k.startswith('module.') else k
        temp[k2] = checkpoint[k]

    net.load_state_dict(temp, strict=True)


def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = '.'.join(k.split('.')[start_idx:])

        new_state_dict[name] = v
    return new_state_dict


def main():
    args = get_args()
    date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
    args.save = args.save + '/' + date_str + '_max_epoch_' + str(args.total_epoch)
    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists(args.save + '/log'):
        os.makedirs(args.save + '/log')
    fh = logging.FileHandler(
        os.path.join(args.save, 'log/train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
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

    args.val_loader_easy = set_data_loader(dataset_attr_word="test_easy", batch_size=1, size=256, shuffle=False,
                                           transforms_compose=val_transforms_compose)
    args.val_loader_hard = set_data_loader(dataset_attr_word="test_hard", batch_size=1, size=256, shuffle=False,
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
        # 去掉分类层
        new_dict = copy_state_dict(pre_train_weight)
        keys = []
        for k, v in new_dict.items():
            if k.startswith('classifier'):  # 将‘’开头的key过滤掉，这里是要去除的层的key
                continue
            keys.append(k)
        new_dict = {k: new_dict[k] for k in keys}
        state_dict = new_dict
        model.load_state_dict(state_dict, strict=False)

    class_names = ['covering', 'device', 'domestic_animal', 'mater', 'person', 'plant', 'structure', 'vertebrate']
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

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    #                                               lambda step: (
    #                                                       1.0 - step / args.total_iters) if step <= args.total_iters else 0,
    #                                               last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    model = model.to(device)

    all_iters = 0
    if args.auto_continue:
        lastest_model, epoch = get_lastest_model()
        if lastest_model is not None:
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint')
            for i in range(epoch):
                scheduler.step()

    args.optimizer = optimizer
    args.loss_function = loss_function
    args.scheduler = scheduler
    total_iters = args.total_epoch * args.train_loader.__len__()

    writer = init_tb_writer_global(args)

    for epoch in range(1, args.total_epoch + 1):
        all_iters = train(model, device, args, epoch, writer=writer, bn_process=True, all_iters=all_iters,
                          total_iters=total_iters)
        validate_easy(model, device, args, epoch, writer=writer)
        validate_hard(model, device, args, epoch, writer=writer)

        save_checkpoint(path=args.save, state={'state_dict': model.state_dict(), }, epoch=epoch,
                        tag='retrain_COME15K_', model_size=args.model_size)


def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV2_Plus")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_resume', type=str, default='./snet_detnas.pkl', help='path for eval model')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--total_epoch', type=int, default=200, help='total epoch')
    parser.add_argument('--learning_rate', type=float, default=1.0, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')

    parser.add_argument('--auto_continue', type=bool, default=False, help='auto continue')
    parser.add_argument('--model_size', type=str, default='Large', choices=['Small', 'Medium', 'Large'],
                        help='size of the model')
    parser.add_argument('--train_dir', type=str, default='data/SOD-SemanticDataset/train',
                        help='path to training dataset')
    parser.add_argument('--val_dir', type=str, default='data/SOD-SemanticDataset/test',
                        help='path to validation dataset')
    parser.add_argument('--fine_tune', type=bool, default=True, help='load pretrain weight at start')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
