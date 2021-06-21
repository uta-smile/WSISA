# WSISA
Implementation of WSISA CVPR 2017
Implemented 4 step:
1. Clustering
``` 
python pca_cluster_img.py
```
Required modification before running the code:
```
data_dir = 'Paht/to/patches' # contains patches for all WSIs (eg 1000pathces/WSI)
```
2. Select clusters

deepConvSurv https://github.com/chunyuan1/deepConvSurv_PyTorch

3. Integration
``` 
python -u main_WSISA_selectedCluster.py | tee -a /path/to/save/log.txt
```
Required modification before running the code:
```
# in file main_WSISA_selectedCluster
selected_cluster = [0, 1, 5]  # contains cluster ID of selected cluster
img_path='path/to/all/patches'
label_path = 'path/to/label/file' # the label file should contains surv and status of each WSI
expand_label_path = 'path/to/extend/label/file'  # the expand label file contains cluster id for each patches
base_path = 'patch/to/trained/model/of/each/cluster'  # trained model in step 2
```
4. Survival prediction

code https://github.com/chunyuan1/WSISA_surv
