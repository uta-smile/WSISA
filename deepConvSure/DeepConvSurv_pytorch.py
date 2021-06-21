from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import numpy as np
import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import lasagne
from lasagne.layers import batch_norm
from lifelines.utils import concordance_index
from lasagne.regularization import regularize_layer_params, l1, l2
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
# import prettytable as pt

from networks import DeepSurv
from networks import NegativeLogLikelihood
from utils import c_index
from utils import adjust_learning_rate


class DeepConvSurv:
    def __init__(self, learning_rate, channel, width, height, lr_decay = 0.01, momentum = 0.9, L2_reg = 0.0, L1_reg = 0.0, standardize = False):
        self.X =  T.ftensor4('x') # patients covariates
        self.E =  T.ivector('e') # the observations vector

        # Default Standardization Values: mean = 0, std = 1
        # self.offset = theano.shared(np.zeros(shape = n_in, dtype=np.float32))
        # self.scale = theano.shared(np.ones(shape = n_in, dtype=np.float32))

    ################################ construct network #############################

        self.l_in = lasagne.layers.InputLayer(
            shape=(None, channel, width, height), input_var=self.X
        )

        self.network = lasagne.layers.Conv2DLayer(
            self.l_in,
            num_filters=32,
            filter_size= 7,
            stride = 3,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),
        )
        self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=(2, 2))

        self.network = lasagne.layers.Conv2DLayer(
            self.network,
            num_filters=32,
            stride = 2,
            filter_size = 5,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),
        )
        self.network = lasagne.layers.Conv2DLayer(
            self.network,
            num_filters=32,
            stride = 2,
            filter_size=3,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),
        )
        self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=(2, 2))

        self.before_output = lasagne.layers.DenseLayer(
            self.network,
            num_units=32,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),
        )
        #self.network = lasagne.layers.DropoutLayer(self.network, p=0.5)

        if standardize:
            network = lasagne.layers.standardize(network, self.offset,
                                                self.scale,
                                                shared_axes = 0)
        self.standardize = standardize

        # Combine Linear to output Log Hazard Ratio - same as Faraggi
        self.network = lasagne.layers.DenseLayer(
            self.before_output, num_units = 1,
            nonlinearity = lasagne.nonlinearities.linear,
            W = lasagne.init.GlorotUniform()
        )

        self.params = lasagne.layers.get_all_params(self.network,
                                                    trainable = True)

        # Relevant Functions
        # self.partial_hazard = T.exp(self.risk(deterministic = True))

        # Set Hyper-parameters:
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.L2_reg = L2_reg
        self.L1_reg = L1_reg
        self.momentum = momentum
        self.channel = channel
        self.width = width
        self.height = height


    def train(self,data_path, label_path, train_index,test_index, valid_index,
              model_index = 0, cluster = 0, num_epochs = 5, batch_size = 10, validation_frequency = 30, patience = 500, improvement_threshold = 0.995, patience_increase=1.2, verbose = True, ratio = 0.8, 
              update_fn = lasagne.updates.nesterov_momentum, **kwargs):
        # torch.cuda.set_device('cuda:3')
        device = torch.device("cuda:2")
        if verbose:
            print('Start training DeepConvSurv')
        #load data and label
        label = pd.read_csv(label_path)
        t = pd.to_numeric(label["surv"], errors='coerce').astype(np.float32)
        e = pd.to_numeric(label["status"], errors='coerce').astype(np.int32)
        t = t.astype("float32").to_numpy()
        e = e.astype("int32").to_numpy()
        imgs = label["img"].values.tolist()
        t_train = t[train_index]
        
        best_param = None
        best_validation_loss = 5000
        done_looping = False
        epoch_num = 0

        # builds network|criterion|optimizer based on configuration
        model = DeepSurv()
        criterion = NegativeLogLikelihood()
        optimizer = eval('optim.Adam')(model.parameters(), lr=self.learning_rate)
        
        # training
        best_c_index = 0
        train_c = None
        flag = 0

        # data transform 
        # MCO
        # mean = [0.82797706, 0.64284584, 0.71804111]
        # std = [0.17938925,0.21496013,0.20454665]
        # NLST mean & std
        mean = [0.69638642, 0.59045078, 0.66915749]
        std = [0.25588675, 0.29427508, 0.24616708]
        self.transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
                
        # start training
        while (epoch_num < num_epochs) and (not done_looping):
            model.to(device)
            criterion.to(device)
            start_time = time.time()
            # iterate over training mini batches and update the weights
            lr = adjust_learning_rate(optimizer, epoch_num,self.learning_rate,self.lr_decay)
            num_batches_train = int(np.ceil(len(t_train) / batch_size))
            # train_losses = []
            
            # train network
            model.train()
            for batch_num in range(num_batches_train):
                batch_slice = slice(batch_size * batch_num,
                                            batch_size * (batch_num + 1))
                batch_index = train_index[batch_slice]
                e_batch = e[batch_index]
                t_batch = t[batch_index]
                # Sort Training Data for Accurate Likelihood
                sort_idx = np.argsort(t_batch)[::-1]
                e_batch = e_batch[sort_idx]
                if sum(e_batch) == 0:
                    # print('pass all 0 status')
                    continue
                t_batch = t_batch[sort_idx]

                img_batch = [imgs[i] for i in batch_index]
                # x_batch = []
                x_batch = torch.FloatTensor()
                for idx in sort_idx:
                    img = img_batch[idx]
                    tmp_img = self.transf(np.array(Image.open(img)))
                    x_batch = torch.cat([x_batch, tmp_img.unsqueeze(0)])
                # image = list(map(lambda trans: trans(x_batch), transf))
                # x_batch = np.asarray(x_batch)
                # x_batch = x_batch.astype(theano.config.floatX)/255.0
                # x_batch = x_batch.reshape(-1, self.channel, self.width, self.height)		
                
                # x_batch = torch.FloatTensor(x_batch).to(device)
                x_batch = x_batch.to(device)
                risk_pred = model(x_batch)
                # print(risk_pred.reshape([-1]))
                t_batch = torch.FloatTensor(t_batch).to(device)
                e_batch = torch.FloatTensor(e_batch).to(device)
                # t_batch = t_batch.to(device)
                # e_batch = e_batch.to(device)
                # risk_pred = risk_pred.cpu()
                # risk_pred = torch.FloatTensor(risk_pred).cuda(non_blocking=True)
                train_loss = criterion(risk_pred, t_batch, e_batch, model)
                if torch.isnan(train_loss):
                    print('train loss nan ')
                # t: survival. e:status
                # if sum(e_batch.data.cpu().numpy())>0:
                #     try:
                #         train_c = c_index(-risk_pred, t_batch, e_batch)
                #     except:
                #         print('e_batch ', e_batch.data.cpu().numpy())
                #         print('risk_pred ', risk_pred.detach().cpu().numpy())
                # updates parameters
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                
                iter = epoch_num * num_batches_train + batch_num
                # validation
                if not valid_index and (iter % validation_frequency == 0):
                    model.eval()
                    x_valid = []
                    #load validation data
                    img_valid = [imgs[i] for i in valid_index]
                    for img in img_valid:
                        x_valid.append(np.array(Image.open(img)))
                    x_valid = np.asarray(x_valid)
                    x_valid = x_valid.astype(theano.config.floatX)/255.0
                    x_valid = x_valid.reshape(-1, self.channel, self.width, self.height)
                    e_valid = e[valid_index]
                    t_valid = t[valid_index]
                    # Sort Validation Data
                    sort_idx = np.argsort(t_valid)[::-1]
                    x_valid = x_valid[sort_idx]
                    e_valid = e_valid[sort_idx]
                    t_valid = t_valid[sort_idx]

                    with torch.no_grad():
                        print('validation')
                        x_valid = torch.FloatTensor(x_valid).to(device)
                        t_valid = torch.FloatTensor(t_valid).to(device)
                        e_valid = torch.FloatTensor(e_valid).to(device)
                        risk_pred = model(x_valid)
                        valid_loss = criterion(risk_pred, t_valid, e_valid, model)
                        valid_c = c_index(-risk_pred, t_valid, e_valid)
                        if best_c_index < valid_c:
                            best_c_index = valid_c
                            flag = 0
                            # saves the best model
                            torch.save({
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'epoch': epoch_num}, os.path.join(models_dir, ini_file.split('\\')[-1]+'.pth'))
                        else:
                            flag += 1
                            if flag >= patience:
                                return best_c_index

            total_time = time.time() - start_time
            print("Epoch: %d, valid_loss=%f, train_loss=%f,  time=%fs"
                % (epoch_num + 1, best_validation_loss, train_loss, total_time))
            # print('\rEpoch: {}\tLoss: {:.8f}({:.8f})\tc-index: {:.8f}({:.8f})\tlr: {:g}'.format(epoch_num, train_loss.item(), valid_loss.item(), train_c, valid_c, lr), end='', flush=False)

            epoch_num = epoch_num + 1
            
        if test_index:
            # ci_test = self.get_concordance_index(imgs,t,e,test_index)
            test_bs=100
            t_test = t[test_index]
            num_batches_test = int(np.ceil(len(t_test) / test_bs))
            hazard_all = torch.FloatTensor().to(device)
            t_all = torch.FloatTensor().to(device)
            e_all = torch.FloatTensor().to(device)
            for batch_num in range(num_batches_test):
                batch_slice = slice(test_bs * batch_num,test_bs * (batch_num + 1))
                batch_index = test_index[batch_slice]
                # if len(batch_index)<100:
                #     import pdb
                #     pdb.set_trace()
                x_test = torch.FloatTensor()
                for img in [imgs[i] for i in batch_index] :
                    x = self.transf(np.array(Image.open(img)))
                    x_test = torch.cat([x_test, x.unsqueeze(0)])
                e_batch = e[batch_index]
                t_batch = t[batch_index]
                e_batch = torch.FloatTensor(e_batch).to(device)
                t_batch = torch.FloatTensor(t_batch).to(device)
                e_all = torch.cat([e_all, e_batch])
                t_all = torch.cat([t_all, t_batch])
                # model.cpu()
                # criterion.cpu()
                model.eval()
                with torch.no_grad():
                    # import pdb
                    # pdb.set_trace()
                    # x_test = x_test.to(device)
                    partial_hazards = model(x_test.to(device))
                    hazard_all = torch.cat([hazard_all, partial_hazards])
                    test_loss = criterion(partial_hazards, t_batch, e_batch, model)
                    if batch_num+1 == num_batches_test:
                        try:
                            ci_test = c_index(-hazard_all, t_all, e_all)
                            # ci_test.append(cur_ci)
                            print(ci_test)
                        except:
                            print('cannot compute c-index ')

        # print("test: ", best_c_index)
        # np.savez(imgmodel_name, *lasagne.layers.get_all_param_values(self.before_output))
        print('test loss ', test_loss)
        print('test: ', ci_test)
        imgmodel_name = './log/wsisa_patch10/convimgmodel_cluster%d_fold%d.pth' %(cluster, model_index)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_num}, imgmodel_name)
                        
        return ci_test


    def load_model(self, params):
        lasagne.layers.set_all_param_values(self.network, params, trainable=True)

