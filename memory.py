'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-17 19:28:55
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2023-01-17 19:31:20
FilePath: /Lingfeng He/xiongyali_new_idea/assignment/memory.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd

class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, cross, features, features_avg, momentum, threshold=0.5, update=0):
        
        ctx.threshold = threshold
        
        ctx.features = features
        ctx.features_avg = features_avg
        ctx.momentum = momentum
        ctx.cross = cross
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
        
        if ctx.cross == False:
            for x, y in zip(inputs, targets):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()
        else:
            batch_size = inputs.shape[0]
            order_list = torch.randperm(batch_size).to(inputs.device)
            for x, y in zip(inputs[order_list], targets[order_list]):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None, None, None, None, None, None


def cm(inputs, indexes, cross, features, features_avg, momentum=0.5, threshold=0.5, update=0):
    return CM.apply(inputs, indexes, cross, features, features_avg, torch.Tensor([momentum]).to(inputs.device), threshold, update)



class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.threshold = 0.5

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('features_avg', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets=None, cross=False):

        inputs = F.normalize(inputs, dim=1)
        
        outputs = cm(inputs, targets, cross, self.features, self.features_avg, self.momentum, threshold=self.threshold)
        outputs /= self.temp

        return outputs
    



class CMWeight(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, weight, features, momentum):
        
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets, weight)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, weight = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
        
        for x, y, w in zip(inputs, targets, weight):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x * w
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None, None, None, None, None, None


def cm_weight(inputs, indexes, cross, features, momentum=0.5):
    return CMWeight.apply(inputs, indexes, cross, features, torch.Tensor([momentum]).to(inputs.device))


class WeightedMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(WeightedMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.gamma = 1.1

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('features_avg', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets=None, weight=None):

        inputs = F.normalize(inputs, dim=1)
        prototypes = self.features[targets]
        sim_mat = (inputs * prototypes).sum(1)
        threshold_mat = torch.zeros_like(sim_mat).to(inputs.device)
        uni_targets = targets.unique()
        for y in uni_targets:
            threshold_mat[y==targets] = sim_mat[y == targets].min().clamp(min=0.) * self.gamma
        weight = torch.as_tensor(sim_mat <= threshold_mat, dtype=torch.float)
        
        outputs = cm_weight(inputs, targets, weight, self.features, self.momentum)
        outputs /= self.temp

        return outputs