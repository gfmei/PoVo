import torch
from torch import nn


def com_tf(lb_list):
    tf_list = [label.sum(dim=0, keepdim=True) for label in lb_list]
    return tf_list


def com_idf(lb_list):
    lb_mean = torch.stack(lb_list, dim=0).mean(dim=1)
    idf = len(lb_list) / (1 + torch.cat(lb_mean, dim=0).sum(dim=0))
    return torch.log(idf)


class FrVot(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, points, features, spt_ids):
        pass

