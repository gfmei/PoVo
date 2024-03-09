import numpy as np
import torch
import torch.distributed as dist


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_vocab_metric(metric_class, base_class_idx, novel_class_idx):
    if isinstance(metric_class, list):
        metric_class = np.array(metric_class)
    metric_base = np.mean(metric_class[base_class_idx])
    metric_novel = np.mean(metric_class[novel_class_idx])
    h_metric = 2 * metric_base * metric_novel / (metric_base + metric_novel + 10e-10)
    m_metric = (metric_base * len(base_class_idx) + metric_novel * len(novel_class_idx)) / (
            len(base_class_idx) + len(novel_class_idx))
    return h_metric, m_metric, metric_base, metric_novel


def intersectionAndUnionGPU(output, target, K, ignore_index=-100):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1).clone()
    target = target.view(-1).clone()
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda(), area_output.cuda()


def update_meter(intersection_meter, union_meter, target_meter, output_meter, preds, labels,
                 n_classes, ignore_label=-100):
    intersection, union, target, output = intersectionAndUnionGPU(
        preds, labels, n_classes, ignore_label
    )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target), dist.all_reduce(output)

    intersection, union, target, output = [intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(),
                                           output.cpu().numpy()]

    intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
    output_meter.update(output)
    # precision = sum(intersection_meter.val) / (sum(output_meter.val) + 1e-10)
    accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

    return intersection_meter, union_meter, target_meter, output_meter, accuracy


def calc_metrics(intersection_meter, union_meter, target_meter, output_meter):
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    precision_class = intersection_meter.sum / (output_meter.sum + 1e-10)
    acc_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    label_ratio_class = output_meter.sum / (sum(output_meter.sum) + 1e-10)
    mIoU = np.mean(iou_class)
    mPre = np.mean(precision_class)
    mAcc = np.mean(acc_class)
    allPre = sum(intersection_meter.sum) / (sum(output_meter.sum) + 1e-10)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    return mIoU, mPre, mAcc, allPre, allAcc, iou_class, precision_class, acc_class, label_ratio_class


def update_binary_acc_meter(intersection_meter, target_meter, binary_preds, labels, idx1, n_classes):
    # idx1: binary_label: 0
    intersection, target = torch.zeros(n_classes).cuda(), torch.zeros(n_classes).cuda()
    binary_idx = [0 if i in idx1 else 1 for i in range(n_classes)]
    binary_idx = np.array(binary_idx)
    for c in range(n_classes):
        intersection[c] = (binary_preds[..., 0][labels == c] == binary_idx[c]).sum()
        target[c] = (labels == c).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(intersection), dist.all_reduce(target)
    intersection, target = intersection.cpu().numpy(), target.cpu().numpy()
    intersection_meter.update(intersection), target_meter.update(target)
    return intersection_meter, target_meter


def calc_binary_acc(intersection_meter, target_meter):
    acc_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mAcc = np.mean(acc_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    return mAcc, allAcc, acc_class


def get_vocab_free_metric(metric_class):
    if isinstance(metric_class, list):
        metric_class = np.array(metric_class)
    metric_all = np.mean(metric_class)
    n_samp = max(metric_class.shape[0], 1e-4)
    m_metric = (metric_all * n_samp) / n_samp
    return m_metric


def cal_vf_metrics(iou_class, acc_class):
    miou = get_vocab_free_metric(iou_class)
    macc = get_vocab_free_metric(acc_class)

    return miou, macc
