from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class Regularization(object):
    def __init__(self, order, weight_decay):
        ''' The initialization of Regularization class
        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        ''' Performs calculates regularization(self.order) loss for model.
        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss

class DeepSurv(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self):
        super(DeepSurv, self).__init__()
        # parses parameters of network from configuration
        # self.drop = config['drop']
        # self.norm = config['norm']
        # self.dims = config['dims']
        # self.activation = config['activation']
        # builds network
        # self.model = self._build_network()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 1)

    # def _build_network(self):
    #     ''' Performs building networks according to parameters'''
    #     layers = []
    #     for i in range(len(self.dims)-1):
    #         if i and self.drop is not None: # adds dropout layer
    #             layers.append(nn.Dropout(self.drop))
    #         # adds linear layer
    #         layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
    #         if self.norm: # adds batchnormalize layer
    #             layers.append(nn.BatchNorm1d(self.dims[i+1]))
    #         # adds activation layer
    #         layers.append(eval('nn.{}()'.format(self.activation)))
    #     # builds sequential network
    #     return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = 0.0
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        mask = torch.ones(y.shape[0], y.shape[0]).to(y.device)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        # neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e)
        l2_loss = self.reg(model)
        # import pdb
        # pdb.set_trace()
        return neg_log_loss + l2_loss