import datetime
import os
import sys
from collections import OrderedDict

import pandas
import torch
import torch.nn as nn
import numpy as np
import PIL
from PIL import Image
import pandas as pd
import time
import logging
import warnings
import argparse
from collections import OrderedDict
from tensorboardX import SummaryWriter
from torchvision.transforms import transforms

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from network import ShuffleNetV2_Plus
from COME15KClassDataset import set_data_loader, OpenCVResize
from tqdm import tqdm
from utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, save_checkpoint_result, \
    get_lastest_model, get_parameters

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


def train_one_epoch(model, device, args, epoch, writer, bn_process=False, all_iters=None, total_iters=None):
    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_loader = args.train_loader

    Top1_err = 0.0
    Top2_err = 0.0
    Top3_err = 0.0
    loss_avg = 0.0
    model.train()
    pbar = tqdm(train_loader, desc='Epoch-train' + str(epoch), unit='batch')
    d_st = time.time()
    df_train_log = pd.DataFrame()
    for iters, (batched_inputs_img, batched_inputs_label) in enumerate(pbar):
        if bn_process:
            adjust_bn_momentum(model, iters + 1)
        all_iters += 1
        labels = batched_inputs_label
        batched_inputs_label = batched_inputs_label.type(torch.LongTensor)
        batched_inputs_img, batched_inputs_label = batched_inputs_img.to(device), batched_inputs_label.to(device)

        output = model(batched_inputs_img)
        loss = loss_function(output, batched_inputs_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prec1, prec2, prec3 = accuracy(output, batched_inputs_label, topk=(1, 2, 3))
        _, preds = torch.max(output, 1)  # 获得当前 batch 所有图像的预测类别
        preds = preds.cpu().numpy()
        labels = labels.detach().cpu().numpy()
        train_loss = loss.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        train_accuracy = accuracy_score(labels, preds)
        log_train_batch = {
            "epoch": epoch,
            "iter": str(all_iters) + "/" + str(total_iters),
            "batch": str(all_iters),
            "lr": scheduler.get_lr()[0],
            "loss_tensor": loss.item(),
            "train_loss_numpy": train_loss,
            "train_accuracy_top1": train_accuracy,
            "prec1": prec1.item() / 100,
            "prec2": prec2.item() / 100,
            "prec3": prec3.item() / 100
        }
        pbar.set_postfix(log_train_batch)
        df_train_log = df_train_log._append(log_train_batch, ignore_index=True)
        Top1_err += 1 - prec1.item() / 100
        Top2_err += 1 - prec2.item() / 100
        Top3_err += 1 - prec3.item() / 100
        loss_avg += loss.item()
    scheduler.step()
    train_time = time.time() - d_st

    printInfo = 'TRAIN epoch {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_lr()[0],
                                                                        loss_avg / iters) + \
                'Top-1 err = {:.6f},\t'.format(Top1_err / iters) + \
                'Top-2 err = {:.6f},\t'.format(Top2_err / iters) + \
                'Top-3 err = {:.6f},\t'.format(Top3_err / iters) + \
                'data_time = {:.6f},\ttrain_time = {:.6f}\n'.format(time.time(),
                                                                    train_time / iters)
    if (epoch == 1):
        logging.info(args.__dict__.__str__())
    logging.info(printInfo)
    print(printInfo)
    hyer_params_dic = args.__dict__
    # hyer_params_dic = {
    #     "epoch_total": args.total_epoch
    # }
    result_dic = {
        "epoch": epoch,
        "lr": scheduler.get_lr()[0],
        "loss": loss_avg / iters,
        "top-1-error": Top1_err / iters,
        "top-2-error": Top2_err / iters,
        "top-3-error": Top3_err / iters
    }
    writer.add_hparams(hyer_params_dic, result_dic, name='train', global_step=epoch)
    writer.add_scalars(main_tag='train', tag_scalar_dict=result_dic, global_step=epoch)

    save_checkpoint(args.save, {
        'state_dict': model.state_dict(),
    }, args.total_epoch, tag='current-')

    return all_iters, df_train_log, result_dic


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
    criterion = nn.CrossEntropyLoss()
    loss_list = []
    labels_list = []
    preds_list = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(pbar_easy):
            labels = target
            labels = labels.to(device)
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
            _, preds = torch.max(output, 1)  # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            test_loss = criterion(output, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            labels = labels.detach().cpu().numpy()
            output = output.detach().cpu().numpy()
            test_loss = test_loss.detach().cpu().numpy()
            test_accuracy = accuracy_score(labels, preds)
            log_test_easy_batch = {
                'process': str(idx) + "/" + str(len(pbar_easy)),
                "epoch": epoch,
                "loss_tensor": loss.item(),
                "test_loss_numpy": test_loss,
                "train_accuracy_top1": test_accuracy,
                "prec1": prec1.item() / 100,
                "prec2": prec2.item() / 100,
                "prec3": prec3.item() / 100
            }
            pbar_easy.set_postfix(log_test_easy_batch, refresh=True)
            loss_list.append(test_loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
        logInfo = 'val-easy Epoch {}: loss = {:.6f},\t'.format(epoch, objs_easy.avg) + \
                  'Top-1 err = {:.6f},\t'.format(1 - top1_easy.avg / 100) + \
                  'Top-2 err = {:.6f},\t'.format(1 - top2_easy.avg / 100) + \
                  'Top-3 err = {:.6f},\t'.format(1 - top3_easy.avg / 100) + \
                  'val_time = {:.6f}\n'.format(time.time() - t1)
        print(logInfo)
        logging.info(logInfo)
        hyer_params_dic = args.__dict__
        result_dic = {
            "epoch": epoch,
            "loss": objs_easy.avg,
            "top-1-error": (1 - top1_easy.avg / 100),
            "top-2-error": (1 - top2_easy.avg / 100),
            "top-3-error": (1 - top3_easy.avg / 100)
        }
        writer.add_hparams(hyer_params_dic, result_dic, name='test_easy', global_step=epoch)
        writer.add_scalars(main_tag='test/easy', tag_scalar_dict=result_dic, global_step=epoch)
    log_test = {}
    log_test['epoch'] = epoch
    # 计算分类评估指标
    log_test['test_easy_loss'] = np.mean(loss_list)
    log_test['test_easy_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_easy_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_easy_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_easy_f1-score'] = f1_score(labels_list, preds_list, average='macro')
    return log_test


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
    criterion = nn.CrossEntropyLoss()
    loss_list = []
    labels_list = []
    preds_list = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(pbar_hard):
            labels = target
            labels = labels.to(device)
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
            _, preds = torch.max(output, 1)  # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            test_loss = criterion(output, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            labels = labels.detach().cpu().numpy()
            output = output.detach().cpu().numpy()
            test_loss = test_loss.detach().cpu().numpy()
            test_accuracy = accuracy_score(labels, preds)
            log_hard_hard_batch = {
                'process': str(idx) + "/" + str(len(pbar_hard)),
                "epoch": epoch,
                "loss_tensor": loss.item(),
                "test_loss_numpy": test_loss,
                "train_accuracy_top1": test_accuracy,
                "prec1": prec1.item() / 100,
                "prec2": prec2.item() / 100,
                "prec3": prec3.item() / 100
            }
            pbar_hard.set_postfix(log_hard_hard_batch, refresh=True)
            loss_list.append(test_loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
        logInfo = 'val-hard Epoch {}: loss = {:.6f},\t'.format(epoch, objs_hard.avg) + \
                  'Top-1 err = {:.6f},\t'.format(1 - top1_hard.avg / 100) + \
                  'Top-2 err = {:.6f},\t'.format(1 - top2_hard.avg / 100) + \
                  'Top-3 err = {:.6f},\t'.format(1 - top3_hard.avg / 100) + \
                  'val_time = {:.6f}\n'.format(time.time() - t1)
        print(logInfo)
        logging.info(logInfo)
        hyer_params_dic = args.__dict__
        result_dic = {
            "epoch": epoch,
            "loss": objs_hard.avg,
            "top-1-error": (1 - top1_hard.avg / 100),
            "top-2-error": (1 - top2_hard.avg / 100),
            "top-3-error": (1 - top3_hard.avg / 100)
        }
        writer.add_hparams(hyer_params_dic, result_dic, name='test_hard', global_step=epoch)
        writer.add_scalars(main_tag='test/hard', tag_scalar_dict=result_dic, global_step=epoch)
    log_test = {}
    log_test['epoch'] = epoch
    # 计算分类评估指标
    log_test['test_hard_loss'] = np.mean(loss_list)
    log_test['test_hard_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_hard_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_hard_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_hard_f1-score'] = f1_score(labels_list, preds_list, average='macro')
    return log_test


def load_checkpoint(net, checkpoint):
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
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # RGB,imageNet1k mean and standard
    ])
    args.train_loader = set_data_loader(dataset_attr_word="train", batch_size=10, size=256, shuffle=True,
                                        transforms_compose=train_transforms_compose,
                                        droup_out_class_label=args.droup_out_class_label)
    assert os.path.exists(args.val_dir)
    val_transforms_compose = transforms.Compose([
        # OpenCVResize(256),
        transforms.Resize(256),
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
    class_names = ['covering', 'device', 'domestic_animal', 'mater', 'person', 'plant', 'structure', 'vertebrate']
    class_names_dic = {1: 'covering', 2: 'device', 3: 'domestic_animal', 4: 'mater', 5: 'person', 6: 'plant',
                       7: 'structure', 8: 'vertebrate'}
    model = ShuffleNetV2_Plus(architecture=architecture, n_class=class_names.__len__(), model_size=args.model_size)
    pre_train_model_weight_dic = {
        'Small': 'shuffle_net_v2_plus_image1K_pretrianed_weight/ShuffleNetV2+.ImageNet1k_pre_trained_Small.pth.tar',
        'Medium': 'shuffle_net_v2_plus_image1K_pretrianed_weight/ShuffleNetV2+.ImageNet1k_pre_trained_Medium.pth.tar',
        'Large': 'shuffle_net_v2_plus_image1K_pretrianed_weight/ShuffleNetV2+.ImageNet1k_pre_trained_Large.pth.tar'
    }
    # shuffleNetV2+ 主干网络分四个stage
    if args.fine_tune:
        layer_dic = {"stage_one": ['0', '1', '2', '3'], "stage_two": ['4', '5', '6', '7'],
                     "stage_three": ['8', '9', '10', '11', '12', '13', '14', '15'],
                     "stage_four": ['16', '17', '18', '19']}
        # 载入四个阶段
        pre_train_weight = torch.load(pre_train_model_weight_dic[args.model_size])
        if args.load_all_pretrain_weight:
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
        else:
            load_pretrain_stage_list = args.load_pretrain_stage
            load_pretrain_layer_list = []
            for load_pretrain_stage_ele in load_pretrain_stage_list:
                for k, v in layer_dic.items():
                    if k == load_pretrain_stage_ele:
                        load_pretrain_layer_list.extend(v)
            new_dict = copy_state_dict(pre_train_weight)
            keys = []
            if type(load_pretrain_layer_list) == list and len(load_pretrain_layer_list) != 0:
                # 只留下指定的层
                for k, v in new_dict.items():
                    if k.startswith('classifier'):  # 将‘’开头的key过滤掉，这里是要去除分类层
                        continue
                    if k.startswith('features'):
                        for ele in load_pretrain_layer_list:
                            if k.startswith('features.' + ele):
                                keys.append(k)
                            else:
                                continue
                    else:
                        keys.append(k)
            else:
                for k, v in new_dict.items():
                    if k.startswith('classifier'):  # 将‘’开头的key过滤掉，这里是要去除的层的key
                        continue
                    keys.append(k)
            new_dict = {k: new_dict[k] for k in keys}
            state_dict = new_dict
            model.load_state_dict(state_dict, strict=False)
        # 载入预训练模型参数后...
        # 冻结部分, 分stage
        frozen_stage_list = args.frozen_stage
        frozen_layer_list = []
        for frozen_stage_ele in frozen_stage_list:
            for k, v in layer_dic.items():
                if k == frozen_stage_ele:
                    frozen_layer_list.extend(v)
        if type(frozen_layer_list) == list and len(frozen_layer_list) != 0:
            for name, value in model.named_parameters():
                if name.startswith('features'):
                    for ele in frozen_layer_list:
                        if name.startswith('features.' + ele):
                            value.requires_grad = False
                        else:
                            continue
        # setup optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=args.weight_decay, amsgrad=False)
    else:
        # 从头训练：随机初始化模型全部权重，从头训练所有层
        optimizer = torch.optim.Adam(get_parameters(model), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=args.weight_decay, amsgrad=False)

    # 更新器
    # optimizer = torch.optim.SGD(get_parameters(model),
    #                             lr=args.learning_rate,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # criterion = CrossEntropyLabelSmooth(8, 0.1)
    # label smooth
    criterion= CrossEntropyLabelSmooth(8, args.label_smooth)
    # criterion = nn.CrossEntropyLoss()
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

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
    total_iters = args.total_epoch * len(args.train_loader)

    writer = init_tb_writer_global(args)
    test_result_dic = {'best_easy_accuracy': 0, 'best_easy_accuracy_epoch': 0,
                       'best_hard_accuracy': 0, 'best_hard_accuracy_epoch': 0,
                       'best_avg_accuracy': 0, 'best_avg_accuracy_epoch': 0}
    df_test_easy_log = pd.DataFrame()
    df_test_hard_log = pd.DataFrame()
    df_train_log_all = pd.DataFrame()
    df_train_log_epoch = pd.DataFrame()
    df_test_final_bset = pd.DataFrame()
    for epoch in range(1, args.total_epoch + 1):
        all_iters, df_train_log, result_dic = train_one_epoch(model, device, args, epoch, writer=writer,
                                                              bn_process=True, all_iters=all_iters,
                                                              total_iters=total_iters)
        df_train_log_epoch = df_train_log_epoch._append(result_dic, ignore_index=True)
        log_easy_test = validate_easy(model, device, args, epoch, writer=writer)
        log_hard_test = validate_hard(model, device, args, epoch, writer=writer)

        df_test_easy_log = df_test_easy_log._append(log_easy_test, ignore_index=True)
        df_test_hard_log = df_test_hard_log._append(log_hard_test, ignore_index=True)
        df_train_log_all = pd.concat([df_train_log_all, df_train_log])
        test_easy_accuracy = log_easy_test['test_easy_accuracy']
        test_hard_accuracy = log_hard_test['test_hard_accuracy']
        test_result_dic = save_checkpoint_result(path=args.save, model=model,
                                                 epoch=epoch,
                                                 tag='retrain_COME15K_', model_size=args.model_size,
                                                 test_easy_accuracy=test_easy_accuracy,
                                                 test_hard_accuracy=test_hard_accuracy, test_result_dic=test_result_dic)
        test_result_dic["epoch"] = epoch
        df_test_final_bset = df_test_final_bset._append(test_result_dic, ignore_index=True)
    df_train_log_all.to_csv(args.save + '/log/' + '训练日志-训练集.csv', index=False)
    df_train_log_epoch.to_csv(args.save + '/log/' + '训练日志-epoch-训练集.csv', index=False)
    df_test_easy_log.to_csv(args.save + '/log/' + '训练日志-easy-测试集.csv', index=False)
    df_test_hard_log.to_csv(args.save + '/log/' + '训练日志-hard-测试集.csv', index=False)
    df_test_final_bset.to_csv(args.save + '/log/' + '训练日志-best-测试集.csv', index=False)
    print(test_result_dic.__str__())


def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV2_Plus")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_resume', type=str, default='./snet_detnas.pkl', help='path for eval model')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--total_epoch', type=int, default=50, help='total epoch')
    parser.add_argument('--learning_rate', type=float, default=0.00003, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--auto_continue', type=bool, default=False, help='auto continue')
    parser.add_argument('--model_size', type=str, default='Medium', choices=['Small', 'Medium', 'Large'],
                        help='size of the model')
    parser.add_argument('--train_dir', type=str, default='data/SOD-SemanticDataset/train',
                        help='path to training dataset')
    parser.add_argument('--val_dir', type=str, default='data/SOD-SemanticDataset/test',
                        help='path to validation dataset')
    parser.add_argument('--fine_tune', type=bool, default=True, help='load pretrain weight at start')
    parser.add_argument('--load_all_pretrain_weight', type=bool, default=True, help='load all pretrain weight at '
                                                                                     'beginning')
    parser.add_argument('--load_pretrain_stage', type=list,
                        default=["stage_one", "stage_two", "stage_three", "stage_four"], help='load pretrain weight '
                                                                                              'at start, working at '
                                                                                              'load_all_pretrain_weight = false')
    parser.add_argument('--frozen_stage', type=list,
                        default=["stage_one"], help='frozen weight')
    parser.add_argument('--droup_out_class_label', type=list,
                        default=[], help='training with drop out the class of images in dataset')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
