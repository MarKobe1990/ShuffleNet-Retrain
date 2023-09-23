import os
import re
import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1, 1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(path, state, epoch, tag='', model_size=''):
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path + "/{}checkpoint-{:06}-{}.pth.tar".format(tag, epoch, model_size))
    torch.save(state, filename)


def save_checkpoint_result(path, state, epoch, tag, model_size, test_easy_accuracy, test_hard_accuracy,
                           test_result_dic):
    if not os.path.exists(path):
        os.makedirs(path)
    # 保存最新的最佳模型文件
    # 1、保存平均最好的
    avg_accuracy_result = (test_easy_accuracy + test_hard_accuracy) / 2
    if avg_accuracy_result > test_result_dic['best_avg_accuracy']:
        # 删除旧的最佳模型文件(如有)
        old_save_file_name = path + "/{}checkpoint-best-avg-{:.3f}-{}.pth.tar".format(tag, test_result_dic[
            'best_avg_accuracy'], model_size)
        if os.path.exists(old_save_file_name):
            os.remove(old_save_file_name)
        # 保存新的最佳模型文件
        new_save_file_name = path + "/{}checkpoint-best-avg-{:.3f}-{}.pth.tar".format(tag, avg_accuracy_result,
                                                                                      model_size)
        print('保存新的平均准确度最高的模型',
              "{}checkpoint-best-avg-{:.3f}-{:06}-{}.pth.tar".format(tag, avg_accuracy_result, epoch, model_size))
        test_result_dic['best_easy_accuracy_epoch'] = epoch
        filename = os.path.join(new_save_file_name)
        torch.save(state, filename)
        test_result_dic['best_avg_accuracy'] = avg_accuracy_result
    # 2、保存test_easy_accuracy最好的
    if test_easy_accuracy > test_result_dic['best_easy_accuracy']:
        # 删除旧的最佳模型文件(如有)
        old_save_file_name = path + "/{}checkpoint-best-easy-{:.3f}-{}.pth.tar".format(tag, test_result_dic[
            'best_easy_accuracy'], model_size)
        if os.path.exists(old_save_file_name):
            os.remove(old_save_file_name)
        # 保存新的最佳模型文件
        new_save_file_name = path + "/{}checkpoint-best-easy-{:.3f}-{}.pth.tar".format(tag, test_easy_accuracy,
                                                                                       model_size)
        print('保存新的easy测试集准确度最高的模型',
              "{}checkpoint-best-easy-{:.3f}-{}.pth.tar".format(tag, test_easy_accuracy, epoch, model_size))
        test_result_dic['best_easy_accuracy'] = epoch
        filename = os.path.join(new_save_file_name)
        torch.save(state, filename)
        test_result_dic['best_easy_accuracy'] = test_easy_accuracy
    # 3、保存test_hard_accuracy最好的
    if test_hard_accuracy > test_result_dic['best_hard_accuracy']:
        # 删除旧的最佳模型文件(如有)
        old_save_file_name = path + "/{}checkpoint-best-hard-{:.3f}-{}.pth.tar".format(tag, test_result_dic[
            'best_hard_accuracy'], model_size)
        if os.path.exists(old_save_file_name):
            os.remove(old_save_file_name)
        # 保存新的最佳模型文件
        new_save_file_name = path + "/{}checkpoint-best-hard-{:.3f}-{}.pth.tar".format(tag,
                                                                                       test_hard_accuracy
                                                                                       , model_size)
        print('保存新的hard测试集准确度最高的模型',
              "{}checkpoint-best-hard-{:.3f}-{}.pth.tar".format(tag, test_hard_accuracy, model_size))
        test_result_dic['best_hard_accuracy'] = epoch
        filename = os.path.join(new_save_file_name)
        torch.save(state, filename)
        test_result_dic['best_hard_accuracy'] = test_hard_accuracy
    return test_result_dic


def get_lastest_model():
    if not os.path.exists('./models'):
        os.mkdir('./models')
    model_list = os.listdir('./models/')
    if model_list == []:
        return None, 0
    model_list.sort()
    lastest_model = model_list[-1]
    iters = re.findall(r'\d+', lastest_model)
    return './models/' + lastest_model, int(iters[0])


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(params=group_no_weight_decay, weight_decay=0.)]
    return groups
