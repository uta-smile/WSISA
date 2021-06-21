#!/usr/bin/env python
'''
Re-inplement of old WSISA with selected cluster. (CVPR 2017)
Implemented as Functional Procedure

running script
python -u main_WSISA_selectedCluster.py | tee -a ./log/clus15_fold5.txt 
'''
import numpy as np
# import sklearn.cross_validation
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
# import DeepConvSurv_pytorch as deep_conv_surv
import pandas as pd
import os
import random
import gc
from PIL import Image
from tqdm import tqdm
from time import time

from utils.WSISA_utils import patient_features
from WSISA_dataloader import WSISA_get_feat_dataloader
from networks import DeepConvSurv

import sklearn
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

#hyperparams
model = 'deepconvsurv'
epochs = 8
lr = 1e-4
seed = 1
batchsize = 64
device = torch.device("cuda:2")

def convert_index(inputpid, expandlabel):
    outputindex = []
    for pid in inputpid:
        tmp = list(expandlabel['pid'][expandlabel['pid']==pid].index)
        outputindex.append(tmp)
    patient_num = [len(x) for x in outputindex]
    outputindex = [y for x in outputindex for y in x]
    return outputindex, patient_num


def tidy_cluster_result(label, index):
        cluster_result = zip(label['pid'][index],
                                   label['img'][index],
                                   label['surv'][index],
                                   label['status'][index],
                                   label['cluster'][index])
        return pd.DataFrame(
            cluster_result,
            columns = ['pid', 'patches', 'surv', 'status', 'cluster'])



def get_patch_features(curr_patch_df, selected_clusters, model_paths, batch_size=256, get_features=True):
    """
    Get features from each model
    :return:
    """
    patch_df = curr_patch_df.copy()

    risk_patch_df = curr_patch_df.copy()

    if get_features:
        fea_dim = 32
        risk_dim = 1

        risk_col_names = ['fea_%d' % i for i in range(risk_dim)]
        column_names = list(risk_patch_df.columns)
        risk_patch_df = risk_patch_df.join(pd.DataFrame(np.zeros((1, risk_dim))))
        risk_patch_df.columns = column_names + risk_col_names

    else:
        fea_dim = 1

    fea_column_names = ['fea_%d'%i for i in range(fea_dim)]
    column_names = list(patch_df.columns)
    patch_df = patch_df.join(pd.DataFrame(np.zeros((1, fea_dim))))
    patch_df.columns = column_names + fea_column_names

    for sc in selected_clusters:
        model_name = model_paths[sc]
        model = DeepConvSurv(get_features=get_features) # set to get features or risks
        # model = nn.DataParallel(model)#.to(device)
        # model.to(device)
        trained_model = torch.load(model_name, map_location=torch.device('cpu'))
        model.load_state_dict(trained_model['model'])


        patches_sc = patch_df[patch_df.cluster == sc]
        print("length of patches in selected cluster %d: " % sc, len(patches_sc))
        num_batches = int(np.ceil(patches_sc.shape[0] / batch_size)) + 1
        print("batch_size", batch_size, "num of batches: ", num_batches)

        patches_path = patches_sc['patches'].tolist()

        model.eval()
        print('patches_path')
        print(len(patches_path))
        # print(patches_path[:5])
        #     # ===================forward=====================
        temp = WSISA_get_feat_dataloader(patches_path, np.arange(len(patches_path)), batch_size=256)

        featloader = temp.get_loader()
        tbar = tqdm(featloader, desc='\r')

        for i_batch, sampled_batch in enumerate(tbar):

            X, id_slice = sampled_batch['feat'], sampled_batch['slice']
            batch_slice = np.asarray(id_slice.cpu().numpy(), dtype=np.int16)
            with torch.no_grad():
                if get_features:
                    feat, risk = model(X)#.to(device))
                else:
                    feat = model(X)#.to(device))  # get risk

            patch_df.loc[patches_sc.index[batch_slice],
            'fea_0':'fea_%d' % (fea_dim - 1)] = feat.cpu().numpy()

            risk_patch_df.loc[patches_sc.index[batch_slice],
            'fea_0':'fea_%d' % (risk_dim - 1)] = risk.cpu().numpy()
            # if i_batch ==3:
            #     break

    return patch_df, risk_patch_df


def train_aggregate(dataset_name, model_index, clusters,
                    train_index, test_index,valid_index, 
                    selected_cluster,each_model_feature_dim,
                    train_cluster_result,test_cluster_result,
                    validation_cluster_result,
                    batch_size = 20, learning_rate = 5e-4):
    """
    calculate the risks/features for ensemble/aggregation

    """
    base_path = '/home/cy/project/deepConvSurv/log/wsisa_patch10/convimgmodel'
    model_paths = ['{}_cluster{}_fold{}.pth'.format(base_path,c,model_index) for c in range(clusters)]


    train_patch_fea, train_patch_risk = get_patch_features(train_cluster_result,
                                        selected_cluster, model_paths,
                                        batch_size = 256, get_features=True)
    # train_patch_fea.to_csv('./results/patches_fea_fold%d.csv' % model_index, index=False)
    training_features = patient_features(train_patch_fea, selected_cluster)
    train_risks = patient_features(train_patch_risk, selected_cluster, fea_dim=1)
    del train_patch_fea
    training_features.to_csv(
        './results/train_patient_%s_features_fold%d_c%d.csv' % (dataset_name, model_index, clusters),
        index=False)
    train_risks.to_csv(
        './results/train_patient_%s_risks_fold%d_c%d.csv' % (dataset_name, model_index, clusters),
        index=False)


    test_patch_fea, test_patch_risk = get_patch_features(test_cluster_result,
                                         selected_cluster, model_paths,
                                         batch_size=256)
    # test_patch_risk = get_patch_features(test_cluster_result, selected_cluster, model_paths, batch_size=256, get_features=False)
    # test_patch_fea.to_csv('patches_fea_fold%d.csv' % model_index, index=False)
    testing_features = patient_features(test_patch_fea,
                                         selected_cluster)
    testing_risks = patient_features(test_patch_risk, selected_cluster, fea_dim=1)
    del test_patch_fea
    testing_features.to_csv(
        './results/test_patient_%s_features_fold%d_c%d.csv' % (dataset_name, model_index, clusters),
        index=False)
    testing_risks.to_csv(
        './results/test_patient_%s_risks_fold%d_c%d.csv' % (dataset_name, model_index, clusters),
        index=False)

    valid_patch_fea, valid_patch_risk = get_patch_features(validation_cluster_result,
                                         selected_cluster, model_paths,
                                         batch_size=64)

    # valid_patch_risk = get_patch_features(validation_cluster_result,
    #                                      selected_cluster, model_paths,
    #                                      batch_size=64, get_features=False)
    # valid_patch_fea.to_csv('patches_fea_fold%d.csv' % model_index, index=False)

    valid_features = patient_features(valid_patch_fea,
                                         selected_cluster)
    valid_risks = patient_features(valid_patch_risk, selected_cluster, fea_dim=1)
    del valid_patch_fea
    valid_features.to_csv(
        './results/validation_patient_%s_features_fold%d_c%d.csv' % (dataset_name, model_index, clusters),
        index=False)

    valid_risks.to_csv(
        './results/validation_patient_%s_risks_fold%d_c%d.csv' % (dataset_name, model_index, clusters),
        index=False)


def train_WSISA(img_path, label_path, expand_label_path, selected_cluster,
                    train_test_ratio, train_valid_ratio, seed=seed,dataset_name='NLST',
                    model='deepconsurv', batchsize=batchsize, epochs=epochs, 
                    lr = lr, cluster_num=10,each_model_feature_dim = 32, **kwargs):
    print(' ')
    print('--------------------- Model Selection ---------------------')
    print('---------------Training Model: ', model, '--------------')
    print('---------------------parameters----------------------------')
    print("epochs: ", epochs, "  tr/test ratio: ", train_test_ratio, "  tr/val ratio: ", train_valid_ratio)
    print("learning rate: ", lr, "batch size: ", batchsize)
    print('-----------------------------------------------------------')
    print(' ' )
    # load labels
    labels = pd.read_csv(label_path)
    expand_label = pd.read_csv(expand_label_path)
    # cluster_id = int(expand_label_path.split('cls')[-1].split('.')[0])
    
    ## generate index
    e = labels["status"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rdn_index = skf.split(np.zeros(len(e)),e)
    testci = []
    index_num = 1
    for trainindex, testindex in rdn_index:
        if index_num < 5:
            print(index_num)
            index_num = index_num + 1
            continue

        test_index, test_patchidxcnt = convert_index(labels['pid'].values[testindex], expand_label)
        
        sss = StratifiedShuffleSplit(n_splits=1,test_size=1-train_valid_ratio, random_state = seed)
        cv_idx = sss.split(np.zeros(len(e.values[trainindex])),e.values[trainindex])
        sublabels = labels['pid'].values[trainindex]
        for tr_idx, val_idx in cv_idx:
            train_index, train_patchidxcnt = convert_index(sublabels[tr_idx], expand_label)
            valid_index, valid_patchidxcnt = convert_index(sublabels[val_idx], expand_label)
        
        # save split
        train_cluster_result = tidy_cluster_result(
                                expand_label,
                                train_index)
        test_cluster_result = tidy_cluster_result(
                                expand_label,
                                test_index)
        validation_cluster_result = tidy_cluster_result(
                                    expand_label,
                                    valid_index)
        cluster_result_name = "%s_cluster_result_fold%d.csv" % (dataset_name,
                                                                index_num)
        print(cluster_result_name)
        train_cluster_result.to_csv('./log/train_'+cluster_result_name, index = False)
        test_cluster_result.to_csv('./log/test_'+cluster_result_name, index = False)
        validation_cluster_result.to_csv('./log/validation_'+cluster_result_name,
                                    index = False)
        

        train_aggregate(dataset_name, index_num, cluster_num,
                        train_index, test_index, valid_index,
                        selected_cluster, each_model_feature_dim,
                        train_cluster_result,
                        test_cluster_result,
                        validation_cluster_result)

 
        # get example shape
        # NLSTimage = Image.open('/home/cy/ssd_4t/nlst/nlst_patch/100012/11445/11445_5352_18745_0693_valid.jpg')
        # NLSTimage = np.array(NLSTimage)
        # width = NLSTimage.shape[0]
        # height = NLSTimage.shape[1]
        # channel = NLSTimage.shape[2]

        # if model=='deepconvsurv':
        #     hyperparams = {
        #     'learning_rate': lr,
        #     'channel': channel,
        #     'width': width,
        #     'height': height,
        #     }
        #     network = deep_conv_surv.DeepConvSurv(**hyperparams)
        #     log = network.train(data_path=img_path, label_path=expand_label_path, train_index = train_index, test_index=valid_index, valid_index = test_index, model_index = index_num, cluster = cluster_id, batch_size = batchsize, ratio = train_test_ratio, num_epochs= epochs)
        #     testci.append(log)
        # else:
        #     print("please select a right model!")
        #     continue

        index_num = index_num + 1

    print("In model: ",model,  " the mean value of test: ", np.mean(testci), "standard value of test: ", np.std(testci))

if __name__ == '__main__':
    print("train_WSISA Unit Test")
    selected_cluster = [1, 2, 5, 6, 8, 9]
    # cluster score [0.4758, 0.5119, 0.5164, 0.4581, 0.4933, 0.5222, 0.5278, 0.4652, 0.5005, 0.5531] 
    root_path = '/home/cy/ssd_4t/nlst/nlst_patch_cluster/nlst_patch10_h20/patchsurv1000_cls'
    for i in [10]:
        print('======================================================')
        print('======================================================')
        print('=================      cluster  %d      ==============' %i)
        print('======================================================')
        print('======================================================')
        train_WSISA(img_path='/smile/nfs/nlst-patch_1000', 
                        label_path = '/home/cy/project/deepConvSurv/validpatients.csv',
                        expand_label_path = '/home/cy/ssd_4t/nlst/nlst_selected_cluster/old_wsisa_all.csv',
                        selected_cluster=selected_cluster, model=model, train_test_ratio=0.9, train_valid_ratio=0.9)
else:
    print("Load Model Selection Module")

    

