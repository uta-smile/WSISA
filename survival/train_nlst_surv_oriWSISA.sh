# survival 属于二分类
python train_nlst_surv_swavfeat.py \
--data_dir 'MCO' \
--ckpt_dir './ckpt_oriWSISA/mco' \
--label_file '5' \
--feat_type 'risks' \
--pool_method 'avgpool' \
--model_choice 'survnet18' \
--gpuID '2' \
--weight_decay 1e-4 \
--num_epochs 500 \
--num_workers 10 \
--batch_size 16 \
--num_sample 20 \
--num_cls 2 \
--use_attention 'none' \
--patch_area 'fg' \
--random_sample 1 \
--summary_name 'mco_surv' \
| tee ./ckpt_oriWSISA/mco_risks_fold5_c10.txt
# | tee ./ckpt/lusc_surv_ebd20_ebd_avgpool.txt

# dataset = 'MCO'
# final_path = '/home/cy/ssd_san/ysong/project/cy/MCO/mco_cluster10_vgg/component40_clus10_p400/'
# all_img_path = '/home/cy/ssd_san/ysong/project/cy/MCO/mco_validpatients.csv'

# dataset = 'LUSC'
# final_path = '/home/cy/ssd_san/ysong/project/cy/LUSC/lusc_cluster10_vgg/component40_clus10_p400/'
# all_img_path = '/home/cy/ssd_san/ysong/project/cy/LUSC/lusc_validpatients_fullpid.csv'

# dataset = 'GBM'
# final_path = '/home/cy/ssd_san/ysong/project/cy/GBM/gbm_cluster10_vgg/component40_clus10_p400/'
# all_img_path = '/home/cy/ssd_san/ysong/project/cy/GBM/gbm_validpatients_vggpid.csv'

# --label_file '/home/cy/ssd_san/ysong/project/cy/LUSC/lusc_selected_cluster/selected_ebd_cls20.csv' \
# --label_file '/home/cy/ssd_san/ysong/project/cy/LUSC/lusc_proto_cluster/embeddings_1000_cls20_.csv' \
# --load_to_memory \
#--resume '/media/newhd/ysong/project/LGP/ckpt/model_best_ep0@20200213_1730.pth.tar' \
#--multi_gpu \
# --data_dir '/media/newhd/ysong/ssd_sumsung/mco_feature1k/resnet50' \
# --ckpt_dir '/media/newhd/ysong/project/TCGA/ckpt_surv' \
# --label_file '/media/titan_3t/mco/MCO_GT/Labels/Clinical/label_json_survival.json' \
# --feat_type 'global_maxpool' global_avgpool\
# --feat_type 'prototype' 'output_x'\
# --pool_method 'avgpool' 'maxpool' 'topk'\
# --data_dir '/home/cy/ssd_san/ysong/project/cy/LUSC/lusc_validpatients.csv' \
# --data_dir '/home/cy/ssd_san/ysong/project/cy/LUSC/lusc_selected_cluster/selected_ebd_cls20_validpatient.csv' \
# --label_file '/home/cy/ssd_san/ysong/project/cy/MCO/mco_selected_cluster/selected_ebd_cls20.csv' \