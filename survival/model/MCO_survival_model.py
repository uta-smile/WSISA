import os
import numpy as np

import torch
import torch.nn as nn
# from resnet import resnet50, Bottleneck, BasicBlock

import sys
sys.path.append('..')
from dataloaders.feature_loader import fm_assemble, index_dataset


class baseline_model2(nn.Module):
    """
    This is the trivial baseline classification model
    """
    def __init__(self,
                 num_cls,
                 reduce_feat='maxpool',
                 use_drop=False,
                 in_channel=2048,
                 num_sample=1000,
                 use_att=True):
        super(baseline_model2, self).__init__()
        self.use_att = use_att
        self.num_cls = num_cls
        self.num_sample = num_sample
        self.reduce_feat = reduce_feat
        print('[Log] Using {} method.'.format(reduce_feat))
        # 2048*num_sample_per_wsi -> 2048*1
        self.global_mp = nn.AdaptiveMaxPool1d(1)
        self.global_ap = nn.AdaptiveAvgPool1d(1)

        # self.maxpool1x1 = nn.AdaptiveMaxPool2d((1, 1))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        out_channel = 512
        self.fc = nn.Linear(in_channel, out_channel)
        self.fc1 = nn.Linear(out_channel, self.num_cls)

        self.bn1 = nn.BatchNorm1d(out_channel)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

        self.attention = nn.Sequential(
            nn.Linear(in_channel, 512),  # V
            nn.Tanh(),
            nn.Linear(512, 1)  # W
        )

        self.att0 = nn.Linear(in_channel, 512)  # V  # tranpose first
        self.att1 = nn.BatchNorm1d(512)  # tranpose first
        self.att2 = nn.Tanh()  # tranpose first
        self.att3 = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.use_att == 'icml':
            print('Use ICML-18 attention')
        elif self.use_att == 'se':
            print('Use SENet attention.')
        elif self.use_att == 'none' or self.use_att == '':
            print('Use trival pooling method.')

    def forward(self, x):
        if self.use_att == 'icml':
            """######ICML 2018: 综合每个patch（instance）的重要性作为attention，将多个inst变成一个############"""
            x = x.transpose(1, 2)  # batch_sizex2048x150 -> batch_sizex150x2048
            Att = self.attention(x)  # batch_sizex150x2048 -> batch_sizex150x1
            Att = self.softmax(Att)  # softmax over bag_size(e.g. 150)
            Att = Att.transpose(1, 2)  # batch_sizex150x1 -> batch_sizex1x150
            # batch_sizex1x150 * batch_sizex150x2048 -> batch_sizex1x2048
            x = torch.bmm(Att, x)
            """old version attention: 等价于icml方法"""
            # Att = self.attention(x.transpose(1, 2))
            # Att = self.attention(x)  # Nx2048x150 -> Nx2048x1
            # Att = self.softmax(Att)  # Nxsoftmax(2048)x1
            # Att = Att.transpose(1, 2)  # Nx2048x1 -> Nx1x2048
            # # wrong:Apply atttenion by matrix product: Nx2048xnum_sample * Nxnum_samplex1 -> Nx2048x1
            # x = torch.bmm(Att,
            #               x)  # right: Nx1x2048 * Nx2048x150 -> Nx1x150
            """#############old version: 衡量每个channel的重要性，而不是每个instance的重要性############"""
            # Att = self.attention(x.transpose(1, 2))  # Nx2048x1000 -> Nx1000x1
            # Att = self.att0(x.transpose(1, 2))
            # Att = self.att1(Att.transpose(1, 2))
            # Att = self.att2(Att.transpose(1, 2))
            # Att = self.att3(Att)
            # Att = self.softmax(Att)
            # x = torch.bmm(x, Att)  # Nx2048xnum_samp * Nxnum_sampx1 -> Nx2048x1
        elif self.use_att == 'se':
            """###########SE version attention#############"""
            Att = self.attention(x.transpose(1, 2))  # att: Nx2048x150->Nx150x1
            Att = self.softmax(Att)  # Nxsoftmax(150)x1
            Att = Att.transpose(1, 2).expand_as(
                x)  # Nx150x1 -> Nx1x150 -> Nx2048x150
            x *= Att  # Nx2048x150 dot Nx2048x150
            x = self.global_mp(x)  # Nx2048x1

        elif self.use_att == 'none' or self.use_att == '':
            if self.reduce_feat == 'maxpool':
                x = self.global_mp(x)
            elif self.reduce_feat == 'avgpool':
                x = self.global_ap(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        # x = self.gn1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc1(x)
        softmax_x = self.softmax(x)  # softmax for AUC
        if self.use_att != 'none':
            return x, softmax_x, Att
        else:
            return x, softmax_x, softmax_x


class end2end_surv(nn.Module):
    """
    Gated_model for survival (MCO)
    """
    def __init__(self,
                 num_cls,
                 reduce_feat='maxpool',
                 use_drop=False,
                 in_channel=2048,
                 num_sample=150,
                 use_att=True):
        super(gated_model_surv, self).__init__()
        self.use_att = use_att
        self.num_cls = num_cls
        self.num_sample = num_sample
        self.reduce_feat = reduce_feat
        print('[Log] Using {} method.'.format(reduce_feat))
        # 2048*num_sample_per_wsi -> 2048*1
        self.global_mp = nn.AdaptiveMaxPool1d(1)
        self.global_ap = nn.AdaptiveAvgPool1d(1)

        out_channel = 512
        self.fc = nn.Linear(2048, 512)
        self.fc1 = nn.Linear(512, 1)  # reg for survival
        # self.fc_risk = nn.Linear(512, num_sample)

        self.bn1 = nn.BatchNorm1d(out_channel)
        # self.bn1 = nn.BatchNorm1d(num_sample)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        # self.softmax = nn.Softmax(dim=1)

        self.feat_len = 512
        self.att_v = nn.Sequential(
            nn.Linear(in_channel, self.feat_len),  # V
            nn.Tanh(),
        )
        self.att_u = nn.Sequential(
            nn.Linear(in_channel, self.feat_len),  # V
            nn.Sigmoid(),
        )
        self.att_weight = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.use_att != 'none':
            print('=> Using gated model!')

    def forward(self, x):
        Att = None
        if self.use_att != 'none':
            """######ICML 2018: 综合每个patch（instance）的重要性作为attention，将多个inst变成一个############"""
            x = x.transpose(1, 2)  # batch_sizex2048x150 -> batch_sizex150x2048
            Att_V = self.att_v(x)  # bsx150x2048 -> bsx150x512
            Att_U = self.att_u(x)  # bsx150x2048 -> bsx150x512
            Att = self.att_weight(
                Att_V *
                Att_U)  # element wise multiplication, bsx150x1-> bsx150x1
            Att = self.softmax(Att)  # softmax over bag_size(e.g. 150)
            Att = Att.transpose(1, 2)  # bsx150x1 -> bsx1x150
            # bsx1x150 * bsx150x2048 -> bsx1x2048
            x = torch.bmm(Att, x)
        else:
            if self.reduce_feat == 'maxpool':
                x = self.global_mp(x)
            elif self.reduce_feat == 'avgpool':
                x = self.global_ap(x)

            # batch_size*2048*num_sample -> bs*ns*2048
            x = x.transpose(1, 2)
        # x = torch.flatten(x, start_dim=1)
        x = self.fc(x)  # bs*num_sample*2048 -> bs*ns*512
        # x = self.gn1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu1(x)
        # x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc1(x)  # x is the risk
        # x = self.fc_risk(x)  # bs*ns*512 -> bs*ns*1 (每个patch产生一个risk)
        return x, Att


class gated_model_surv0(nn.Module):
    """
    Gated_model for survival (MCO)
    """
    def __init__(self,
                 num_cls,
                 reduce_feat='maxpool',
                 use_drop=False,
                 in_channel=2048,
                 num_sample=150,
                 use_att=True):
        super(gated_model_surv, self).__init__()
        self.use_att = use_att
        self.num_cls = num_cls
        self.num_sample = num_sample
        self.reduce_feat = reduce_feat
        print('[Log] Using {} method.'.format(reduce_feat))
        # 2048*num_sample_per_wsi -> 2048*1
        self.global_mp = nn.AdaptiveMaxPool1d(1)
        self.global_ap = nn.AdaptiveAvgPool1d(1)

        out_channel = 512
        self.fc = nn.Linear(2048, 512)
        self.fc1 = nn.Linear(512, 1)  # reg for survival
        # self.fc_risk = nn.Linear(512, num_sample)

        self.bn1 = nn.BatchNorm1d(out_channel)
        # self.bn1 = nn.BatchNorm1d(num_sample)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        # self.softmax = nn.Softmax(dim=1)

        self.feat_len = 512
        self.att_v = nn.Sequential(
            nn.Linear(in_channel, self.feat_len),  # V
            nn.Tanh(),
        )
        self.att_u = nn.Sequential(
            nn.Linear(in_channel, self.feat_len),  # V
            nn.Sigmoid(),
        )
        self.att_weight = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.use_att != 'none':
            print('=> Using gated model!')

    def forward(self, x):
        Att = None
        # import pdb; pdb.set_trace()
        if self.use_att != 'none':
            """######ICML 2018: 综合每个patch（instance）的重要性作为attention，将多个inst变成一个############"""
            x = x.transpose(1, 2)  # batch_sizex2048x150 -> batch_sizex150x2048
            Att_V = self.att_v(x)  # bsx150x2048 -> bsx150x512
            Att_U = self.att_u(x)  # bsx150x2048 -> bsx150x512
            Att = self.att_weight(
                Att_V *
                Att_U)  # element wise multiplication, bsx150x1-> bsx150x1
            Att = self.softmax(Att)  # softmax over bag_size(e.g. 150)
            Att = Att.transpose(1, 2)  # bsx150x1 -> bsx1x150
            # bsx1x150 * bsx150x2048 -> bsx1x2048
            x = torch.bmm(Att, x)
        else:
            # print('x shape before reduce feat  ', x.shape)
            if self.reduce_feat == 'maxpool':
                x = x.transpose(1, 2)
                x = self.global_mp(x)
            elif self.reduce_feat == 'avgpool':
                x = x.transpose(1, 2)
                x = self.global_ap(x)

            # batch_size*2048*num_sample -> bs*ns*2048
            # print('x shape before transpose  ', x.shape)
            x = x.transpose(1, 2)
            # print('x shape before fc  ', x.shape)
        # x = torch.flatten(x, start_dim=1)
        x = self.fc(x)  # bs*num_sample*2048 -> bs*ns*512
        # x = self.gn1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu1(x)
        # x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc1(x)  # x is the risk
        # x = self.fc_risk(x)  # bs*ns*512 -> bs*ns*1 (每个patch产生一个risk)
        return x, Att


class gated_model_surv(nn.Module):
    """
    Gated_model for survival (MCO)
    """
    def __init__(self,
                 num_cls,
                 reduce_feat='maxpool',
                 use_drop=False,
                 in_channel=2048,
                 num_sample=150,
                 use_att=True):
        super(gated_model_surv, self).__init__()
        self.use_att = use_att
        self.num_cls = num_cls
        self.num_sample = num_sample
        self.reduce_feat = reduce_feat
        print('[Log] Using {} method.'.format(reduce_feat))
        # 2048*num_sample_per_wsi -> 2048*1
        self.global_mp = nn.AdaptiveMaxPool1d(1)
        self.global_ap = nn.AdaptiveAvgPool1d(1)

        out_channel = 32
        self.fc = nn.Linear(6, 32)
        self.fc1 = nn.Linear(32, 1)  # reg for survival
        # self.fc_risk = nn.Linear(512, num_sample)

        self.bn1 = nn.BatchNorm1d(out_channel)
        # self.bn1 = nn.BatchNorm1d(num_sample)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        # self.softmax = nn.Softmax(dim=1)

        self.feat_len = 512
        self.att_v = nn.Sequential(
            nn.Linear(in_channel, self.feat_len),  # V
            nn.Tanh(),
        )
        self.att_u = nn.Sequential(
            nn.Linear(in_channel, self.feat_len),  # V
            nn.Sigmoid(),
        )
        self.att_weight = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.use_att != 'none':
            print('=> Using gated model!')

    def forward(self, x):
        Att = None
        # import pdb; pdb.set_trace()
        # if self.use_att != 'none':
        #     """######ICML 2018: 综合每个patch（instance）的重要性作为attention，将多个inst变成一个############"""
        #     x = x.transpose(1, 2)  # batch_sizex2048x150 -> batch_sizex150x2048
        #     Att_V = self.att_v(x)  # bsx150x2048 -> bsx150x512
        #     Att_U = self.att_u(x)  # bsx150x2048 -> bsx150x512
        #     Att = self.att_weight(
        #         Att_V *
        #         Att_U)  # element wise multiplication, bsx150x1-> bsx150x1
        #     Att = self.softmax(Att)  # softmax over bag_size(e.g. 150)
        #     Att = Att.transpose(1, 2)  # bsx150x1 -> bsx1x150
        #     # bsx1x150 * bsx150x2048 -> bsx1x2048
        #     x = torch.bmm(Att, x)
        # else:
            # if self.reduce_feat == 'maxpool':
            #     x = self.global_mp(x)
            # elif self.reduce_feat == 'avgpool':
            #     x = self.global_ap(x)

            # batch_size*2048*num_sample -> bs*ns*2048
            # x = x.transpose(1, 2)
        # x = torch.flatten(x, start_dim=1)
        x = self.fc(x)  # bs*num_sample*2048 -> bs*ns*512
        # x = self.gn1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc1(x)  # x is the risk
        # x = self.fc_risk(x)  # bs*ns*512 -> bs*ns*1 (每个patch产生一个risk)
        return x, Att



if __name__ == '__main__':
    # unit test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = baseline_model(num_cls=4)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    data_dir = '/media/titan_3t/mco/mco_patch_feature/feature1000'
    groups = os.listdir(data_dir)
    # print(len(groups))
    dataset_dict = index_dataset(groups, data_dir)
    # print(dataset_dict)
    for k, v in dataset_dict.items():
        if len(v) != 1000:
            print(k)
            print(len(v))

    tmp_key = list(dataset_dict.keys())[0]
    big_featmap = fm_assemble(dataset_dict[tmp_key], num_sample=100)
    print(big_featmap.shape)

    big_featmap = torch.FloatTensor([big_featmap]).to(device)
    out = model(big_featmap)

    import pdb
    pdb.set_trace()

    loss = criterion(out, big_featmap.long())
