import os
import shutil
import numpy as np
import time
import argparse
from tqdm import tqdm
import pdb
from pprint import pprint

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

# from sklearn.preprocessing import label_binarize

# from dataloaders.feature_loader_surv import PatchFeature
# from dataloaders.NLSTLoader import PatchFeature
# from dataloaders.NLSTLoader_cluster import PatchFeature
from dataloaders.NLSTLoader_wsisaFeat import PatchFeature
# from model.cls_baseline_model import baseline_model
from model.MCO_survival_model import baseline_model2, gated_model_surv
# from model.NLST_survival_model_selectedcluster import baseline_model2, gated_model_surv
from utils.helper import accuracy, save_checkpoint, format_time
from utils.surv_utils import accuracy_cox, accuracy, concordance_index, cox_log_rank, CIndex_lifeline
from utils.helper import plot_confusion_matrix, plot_roc_curve_and_compute_auc
from utils.helper import inplace_relu

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    type=str,
    default='/media/titan_3t/mco/mco_patch_feature/feature1000',
    help='path to dataset (or feature files)')
parser.add_argument('--label_file',
                    type=str,
                    default="/media/ssd/all_mco_with_label.json",
                    help='path to the label file')
parser.add_argument('--summary_name',
                    type=str,
                    default='',
                    help='path to the summary files')
parser.add_argument('--ckpt_dir',
                    type=str,
                    default="./ckpt/nlst_surv",
                    help='path to store trained model files')
parser.add_argument('--model_choice',
                    type=str,
                    default="no_gate",
                    help='Model to choose: no_gate/gate/conv_gate')
parser.add_argument('--pool_method',
                    type=str,
                    default="maxpool",
                    help='Pooling method')
parser.add_argument('--feat_type',
                    type=str,
                    default="global_maxpool",
                    help='Feature type (global_maxpool/global_avgpool)')
parser.add_argument('--save_model_freq',
                    type=int,
                    default=200,
                    help='How often (how many epochs) to save model')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpuID',
                    default='0,1,2,3',
                    type=str,
                    help='GPU id to use.')
# parser.add_argument('--gpuid', default='0,1,2,3', type=str, help='GPU id to use.')
parser.add_argument('--multi_gpu',
                    default=False,
                    action='store_true',
                    help='Whether to multi-gpu')
parser.add_argument('--load_to_memory',
                    default=False,
                    action='store_true',
                    help='Whether load all data into memory')
parser.add_argument('--use_attention',
                    default='icml',
                    type=str,
                    help='Attention method to use, icml/icml_gated/se/none(or '
                    ')')
# parser.add_argument('--use_attention',
#                     default=False,
#                     action='store_true',
#                     help='Whether to use attention in model.')
parser.add_argument('--random_sample',
                    type=int,
                    default=0,
                    help='Whether random sample patches in bag.')
parser.add_argument('--patch_area',
                    default='fg',
                    type=str,
                    help='Foreground (fg) or background (bg)')
parser.add_argument('--num_epochs',
                    type=int,
                    default=100,
                    help='number of total epochs to run')
parser.add_argument('--num_sample',
                    type=int,
                    default=400,
                    help='patches(must have int sqrt) sampled from total 1k')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--num_cls',
                    type=int,
                    default=6,
                    help='Total num of classes')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--cuda', action="store_true", default=True)
parser.add_argument('--print_freq',
                    type=int,
                    default=1,
                    help='freq of printing log info')
parser.add_argument('--compute_cm_freq',
                    type=int,
                    default=20,
                    help='frequency of computing confusion matrix')
parser.add_argument('--avg_loss_range',
                    type=int,
                    default=100,
                    help='Step range used to average the loss')
parser.add_argument('--topk_risk', type=int, default=5)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.manual_seed(42)
best_acc_val = 0.
best_cindex_val = 0.


def train_model(train_loader,
                model,
                device,
                criterion,
                optimizer,
                args,
                writer,
                epoch=0,
                data_len=0,
                all_labels=[0, 1, 2, 3, 4, 5]):
    """Args:
            data_len: length of dataset
    """
    model.train()

    # all_pred_label = torch.FloatTensor().to(device)  # argmax result
    # all_pred_prob = torch.FloatTensor().to(device)  # for ROC&AUC
    # all_gt = torch.FloatTensor().to(device)

    pred_risk_all = torch.FloatTensor().to(device)
    surv_time_all = torch.FloatTensor().to(device)
    status_all = torch.FloatTensor().to(device)
    # model.to(device)
    # print('Total training data:{}'.format(data_len))
    start_time = time.time()
    avg_acc_train = np.array([])
    # avg_loss = 0.0
    epoch_loss = 0.0
    # num_sample = 0
    for step, (feat, surv_time, vital_status, fn) in enumerate(
            tqdm(train_loader,
                 dynamic_ncols=True,
                 desc='[train {}/{}] {}smaples'.format(epoch, args.num_epochs,
                                                       data_len))):
        # import pdb
        # pdb.set_trace()
        if not vital_status.byte().any():
            continue  # if all zeros, no event occured, skip
        feat, surv_time, vital_status = feat.to(
            device), surv_time.to(
                device), vital_status.to(device)
        optimizer.zero_grad()
        #break
        """=====================forward: compute output======================"""
        pred_risk, att_tensor = model(feat)  # risk: [num_sample,1]
        # average the topk patches' risk of each WSI
        # comment if using avgpool or maxpool
        if args.pool_method == 'topk':
            pred_risk = pred_risk.topk(args.topk_risk, dim=1)[0]
            pred_risk = pred_risk[:, 1:, ...].mean(dim=1)

        # pdb.set_trace()  # check shape of surv_time, make it n*1
        surv_time = surv_time.view(-1, 1)
        patient_len = surv_time.shape[0]
        R_matrix = np.zeros([patient_len, patient_len], dtype=int)
        for i in range(patient_len):
            for j in range(patient_len):
                R_matrix[i, j] = surv_time[j] >= surv_time[i]

        R_matrix = torch.FloatTensor(R_matrix).to(device)
        y_status = vital_status.float()

        theta_pred = pred_risk.reshape(-1)
        exp_theta_pred = torch.exp(theta_pred)

        loss_nll = -torch.mean((theta_pred - torch.log(
            torch.sum(exp_theta_pred * R_matrix, dim=1))) * y_status)

        l1_norm = 0.
        for W in model.parameters():
            # torch.abs(W).sum() is equivalent to W.norm(1)
            l1_norm += torch.abs(W).sum()
        loss = loss_nll + args.weight_decay * l1_norm


        """=========back prob: compute gradient and do SGD/Adam step========="""
        loss.backward()
        optimizer.step()

        # metrics log
        surv_time_all = torch.cat([surv_time_all, surv_time])
        status_all = torch.cat([status_all, y_status])
        pred_risk_all = torch.cat([pred_risk_all, pred_risk])

        acc_step = accuracy_cox(pred_risk.data, y_status)
        # import pdb; pdb.set_trace()
        p_value = cox_log_rank(pred_risk_all.data, status_all, surv_time_all)
        try:
            c_index = CIndex_lifeline(pred_risk_all.data, status_all, surv_time_all)
        except:
            # print('train === c index unpair ', len(status_all.data.cpu().numpy().reshape(-1)))
            c_index = -1
        
        # Note: 不除batch_size就会出现锯齿状loss
        gstep = epoch * data_len / args.batch_size + step
        # avg_loss += loss.item()
        epoch_loss += loss.item()
        writer.add_scalar('Accuracy/train_step', acc_step, global_step=gstep)
        writer.flush()
        #         model4sum, train_data_sample.to(device))  # model graph, with input
        # if step % args.print_freq == 0:
        # total_time = time.time() - start_time
        # h_total, m_total, sec_total = format_time(total_time)
        # h_avg, m_avg, sec_avg = format_time(total_time / (step + 1))

    print('[train {}/{}] {} -'.format(epoch, args.num_epochs, step),
          'Loss: {:.4f} -'.format(epoch_loss / (step + 1)),
          'Cumulative_Acc: {:.4f} -'.format(acc_step))
          #'C-index:{:.4f}, p-value: {:.4f}'.format(c_index, p_value))

    # epoch-wise log
    epoch_loss /= (step + 1)
    acc_avg = accuracy_cox(pred_risk_all.data, status_all)
    writer.add_scalar('Accuracy/train_epoch', acc_avg, global_step=epoch)
    writer.add_scalar('Loss/train_epoch', epoch_loss, global_step=epoch)
    writer.add_scalar('P_Value/train', p_value, global_step=epoch)
    #writer.add_scalar('C_Index/train', c_index, global_step=epoch)
    # import pdb
    # pdb.set_trace()
    # plot_confusion_matrix(
    #     status_all.cpu().numpy(),
    #     torch.argmax(pred_risk_all.data.cpu().detach().numpy(), 1),
    #     writer,
    #     phase='train',
    #     epoch=epoch,
    #     labels=all_labels,
    # )
    # auc_micro, auc_macro = plot_roc_curve_and_compute_auc(
    #     all_gt.cpu().numpy(),
    #     all_pred_prob.cpu().detach().numpy(),
    #     writer,
    #     phase='train',
    #     epoch=epoch,
    #     labels=all_labels)
    # writer.add_scalar('AUC_Micro/train', auc_micro, epoch)
    # writer.add_scalar('AUC_Macro/train', auc_macro, epoch)
    return epoch_loss, acc_avg, c_index, p_value  #, auc_micro, auc_macro


def test_model(test_loader,
               model,
               device,
               criterion,
               args,
               writer,
               epoch,
               data_len,
               phase='val',
               all_labels=[0, 1, 2, 3, 4, 5]):
    """Args:
        data_len: length of dataset
        phase: validation or test phase

    """
    model.eval()  # influence bn/dropout layers
    # model.to(device)

    pred_risk_all = torch.FloatTensor().to(device)
    surv_time_all = torch.FloatTensor().to(device)
    status_all = torch.FloatTensor().to(device)
    # model.to(device)
    # print('Total training data:{}'.format(data_len))
    start_time = time.time()
    avg_acc_train = np.array([])
    epoch_loss = 0.0

    with torch.no_grad():  # skip gradient computation and BP (for speeding up)
        for step, (feat, surv_time, vital_status,
                   fn) in enumerate(
                       tqdm(test_loader,
                            desc='[{} {}/{}] {} smaples'.format(
                                phase, epoch, args.num_epochs, data_len),
                            dynamic_ncols=True)):

            if not vital_status.byte().any():
                print('========== No events in this batch, skip ==========')
                continue  # if all zeros, no event occured, skip
            feat, surv_time, vital_status = feat.to(
                device), surv_time.to(
                    device), vital_status.to(device)
            """=====================forward: compute output======================"""
            pred_risk, att_tensor = model(feat)
            if args.pool_method == 'topk':
                pred_risk = pred_risk.topk(args.topk_risk, dim=1)[0]
                pred_risk = pred_risk[:, 1:, ...].mean(dim=1)
            #print(pred_risk)

            # pdb.set_trace()  # check shape of surv_time, make it n*1
            surv_time = surv_time.view(-1, 1)
            patient_len = surv_time.shape[0]
            R_matrix = np.zeros([patient_len, patient_len], dtype=int)
            for i in range(patient_len):
                for j in range(patient_len):
                    R_matrix[i, j] = surv_time[j] >= surv_time[i]

            R_matrix = torch.FloatTensor(R_matrix).to(device)
            y_status = vital_status.float()

            theta_pred = pred_risk.reshape(-1)
            exp_theta_pred = torch.exp(theta_pred)

            loss_nll = -torch.mean((theta_pred - torch.log(
                torch.sum(exp_theta_pred * R_matrix, dim=1))) * y_status)

            l1_norm = 0.
            for W in model.parameters():
                # torch.abs(W).sum() is equivalent to W.norm(1)
                l1_norm += torch.abs(W).sum()
            loss = loss_nll + args.weight_decay * l1_norm

            # metrics log
            surv_time_all = torch.cat([surv_time_all, surv_time])
            status_all = torch.cat([status_all, y_status])
            pred_risk_all = torch.cat([pred_risk_all, pred_risk])

            acc_step = accuracy_cox(pred_risk.data, y_status)
            p_value = cox_log_rank(pred_risk_all.data, status_all,
                                   surv_time_all)
            try:
                c_index = CIndex_lifeline(pred_risk_all.data, status_all,surv_time_all)
            except:
                # print('test === c index unpair ', len(status_all.data.cpu().numpy().reshape(-1)))
                c_index = -1

            epoch_loss += loss.item()
            # Note: 不除batch_size就会出现锯齿状loss
            gstep = epoch * data_len / args.batch_size + step
            writer.add_scalar('Loss/{}_step'.format(phase),
                              loss,
                              global_step=gstep)
            writer.add_scalar('Accuracy/{}_step'.format(phase),
                              acc_step,
                              global_step=gstep)
            writer.flush()

        print('[{} {}/{}] - {}'.format(phase, epoch, args.num_epochs, step),
              'Loss: {:.4f} -'.format(epoch_loss / (step + 1)),
              'Cumulative_Acc: {:.4f} -'.format(acc_step),
              'C-index:{:.4f}, P-value: {:.4f}'.format(c_index, p_value))

        # epoch-wise log
        epoch_loss /= (step + 1)
        acc_avg = accuracy_cox(pred_risk_all.data, status_all)
        writer.add_scalar('Accuracy/{}_epoch'.format(phase),
                          acc_avg,
                          global_step=epoch)
        writer.add_scalar('Loss/{}_epoch'.format(phase),
                          epoch_loss,
                          global_step=epoch)
        writer.add_scalar('P_Value/{}'.format(phase),
                          p_value,
                          global_step=epoch)
        writer.add_scalar('C_Index/{}'.format(phase),
                          c_index,
                          global_step=epoch)
    return epoch_loss, acc_avg, c_index, p_value  #, auc_micro, auc_macro


def main():
    global best_acc_val
    global best_cindex_val
    args = parser.parse_args()
    pprint(vars(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuID

    all_labels = list(range(args.num_cls))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if args.model_choice == 'gated':
    #     model_arch = gated_model_surv
    # elif args.model_choice == 'no_gate':
    #     model_arch = vim

    # gated_model or baseline_model2
    model = gated_model_surv(num_cls=args.num_cls,
                       num_sample=args.num_sample,
                       reduce_feat=args.pool_method,
                       use_att=args.use_attention)
    # model = baseline_model3(num_cls=args.num_cls)
    model4sum = model
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        print("[Log] Use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
        model4sum = model.module
    model.to(device)

    # weights = torch.Tensor([0.42, 1.0, 0.65, 0.4, 0.68, 0.3])
    weights = None
    criterion = nn.CrossEntropyLoss(weight=weights,
                                    reduction='mean').to(device)
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             args.lr,
    #                             momentum=0.9,
    #                             nesterov=True,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer,
                                  factor=0.5,
                                  patience=10,
                                  mode='min')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpuID is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpuID)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc_val = checkpoint['best_acc_val']
            best_cindex_val = checkpoint['best_cindex_val']
            # if args.gpuID is not None:
            #     # best_acc_val may be from a checkpoint from a different GPU
            #     resume_gpuid = checkpoint['gpuID']
            #     best_acc_val = best_acc_val.to(resume_gpuid)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print("Setup train/val/test dataset loader...")
    dataset_train = PatchFeature(num_sample=args.num_sample,
                                 data_dir=args.data_dir,
                                 label_file=args.label_file,
                                 split='train',
                                 feat_type=args.feat_type,
                                 random_sample=args.random_sample,
                                 patch_area=args.patch_area,
                                 load_into_memory=args.load_to_memory)
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=False)

    dataset_val = PatchFeature(num_sample=args.num_sample,
                               data_dir=args.data_dir,
                               label_file=args.label_file,
                               split='val',
                               feat_type=args.feat_type,
                               random_sample=args.random_sample,
                               patch_area=args.patch_area,
                               load_into_memory=args.load_to_memory)
    dataloader_val = DataLoader(dataset=dataset_val,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False)

    dataset_test = PatchFeature(num_sample=args.num_sample,
                                data_dir=args.data_dir,
                                label_file=args.label_file,
                                split='test',
                                feat_type=args.feat_type,
                                random_sample=args.random_sample,
                                patch_area=args.patch_area,
                                load_into_memory=args.load_to_memory)
    dataloader_test = DataLoader(dataset=dataset_test,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)

    print("Setup SummaryWriter...")
    # setup the summary writer
    # train_data_sample = torch.unsqueeze(train_data_sample, -1)
    writer = SummaryWriter(comment='_' + args.summary_name + '_' +
                           args.model_choice)

    # with writer:
    #     train_data_sample, lbs, fns = iter(dataloader_train).next()
    #     writer.add_graph(
    #         model4sum, train_data_sample.to(device))  # model graph, with input

    print('Start train/val/test...')
    for ep in range(args.start_epoch, args.num_epochs):
        print("\n=> Epoch[{}] | Taining on [{}] smaples".format(
            ep, dataset_train.__len__()))
        loss_avg_train, avg_acc_train, c_index, p_value = train_model(
            dataloader_train,
            model,
            device,
            criterion,
            optimizer,
            args,
            writer,
            epoch=ep,
            data_len=dataset_train.__len__(),
            all_labels=all_labels,
        )

        # print('\n=> Epoch[{}] | Evaluating on [{}] samples'.format(
        #     ep, dataset_val.__len__()))
        loss_avg_val, avg_acc_val, c_index, p_value = test_model(
            dataloader_val,
            model,
            device,
            criterion,
            args,
            writer,
            ep,
            dataset_val.__len__(),
            phase='val',
            all_labels=all_labels,
        )
        scheduler.step(loss_avg_val)
        print(
            '[val {0}/{5}] Loss_avg: {4:.4f} | Acc_avg: {1:.4f} | C-index: {2:.4f} | P-value:{3:.4f}'
            .format(ep, avg_acc_val, c_index, p_value, loss_avg_val,
                    args.num_epochs))

        # remember best acc@val and save checkpoint
        # is_best = avg_acc_val >= best_acc_val
        # best_acc_val = max(avg_acc_val, best_acc_val)
        is_best = c_index > best_cindex_val
        best_cindex_val = max(c_index, best_cindex_val)

        if ep % args.save_model_freq == 0 or is_best:
            save_checkpoint(
                {
                    'epoch': ep,
                    'state_dict': model.state_dict(),
                    'best_acc_val': best_acc_val,
                    'best_cindex_val': best_cindex_val,
                    'optimizer': optimizer.state_dict(),
                    'gpuID': args.gpuID,
                },
                is_best,
                filename='checkpoint@ep{}.pth.tar'.format(ep),
                dst_path=args.ckpt_dir)

        loss_avg_test, avg_acc_test, c_index, p_value = test_model(
            dataloader_test,
            model,
            device,
            criterion,
            args,
            writer,
            ep,
            dataset_test.__len__(),
            phase='test',
            all_labels=all_labels,
        )
        print(
            '[test {0}/{5}] Loss_avg: {4:.4f} | Acc_avg: {1:.4f} | C-index: {2:.4f} | P-value:{3:.4f}'
            .format(ep, avg_acc_test, c_index, p_value, loss_avg_test,
                    args.num_epochs))
        if is_best:
            print(
                '\n=> Test C-index @ Val_best_C-index: {:.4f}'.format(c_index))

    writer.close()


if __name__ == '__main__':
    main()
