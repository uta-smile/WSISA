"""
Contain Codes for WSISA

"""
import pandas as pd
from time import time
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
from model import DeepConvSurv
import numpy as np
import os
import cv2
from PIL import Image
from tqdm import tqdm
def tidy_cluster_result(kmeans_labels, label, index):
        cluster_result = zip(label['pid'][index],
                                   label['patches'][index],
                                   label['surv'][index],
                                   label['status'][index],
                                   list(kmeans_labels))
        return pd.DataFrame(
            cluster_result,
            columns = ['pid', 'patches', 'surv', 'status', 'cluster'])

def kmeansClusterFit_path(root_dir, path, kmeans, batch_size=256):

    num_batches = int(np.ceil(path.shape[0] / batch_size)) + 1

    pred_cluster_id = np.zeros(len(path))

    for i in tqdm(range(num_batches)):
        batch_slice = slice(batch_size * i,
                            batch_size * (i + 1))
        img_batch = path[batch_slice]
        x_batch = []
        for img_path in img_batch:

            svs_name = img_path.split('_')[0]
            full_path = os.path.join(root_dir, svs_name, img_path)

            thumbnail = Image.open(full_path)
            thumbnail = np.array(thumbnail)

            # convert a 2D image to a vector
            thumbnail = thumbnail.ravel().tolist()


            x_batch.append(thumbnail)
        try:
            x_batch = np.asarray(x_batch)
        # ===================forward=====================

            pred = kmeans.predict(x_batch)
            pred_cluster_id[batch_slice] = pred
        except:
            continue


    return pred_cluster_id







def kmeansClusterFit(data, kmeans):
            #input: (IDs, features)
            #output: (IDs, cluster, centroids)
            pred = kmeans.predict(data)
            return pred

def kmeansClustering(data, n_cluster):
            #input: (IDs, features)
            #output: (IDs, cluster, centroids)
            kmeans = KMeans(n_clusters = n_cluster).fit(data)
            return kmeans

def get_cluster_index(kmeansresult, c):
        outputindex = []
        outputindex = [i for i in range(len(list(kmeansresult)))
                     if list(kmeansresult)[i]==c ]
        return outputindex



def patient_features(patch_df, selected_clusters, fea_dim = 32):
        """Return patient-wise features given selected clusters and models

        It returns patient-wise features via aggregating the features of each
        separate patches.

        Args:
            patch_df: a pandas dataframe which contains columns of pid,
                patch_fea, cluster,etc.
            selected_clusters: a list indicates which clusters have been selected
            fea_dim: feature dimensions for each cluster

        Returns:
            patient_feas: a pandas dataframe contains: pid, surv, status and feas
        """
        patients = patch_df['pid'].unique().tolist()
        pid = []
        surv = []
        status = []
        features = []

        for p in patients:
            pid.append(p)
            surv.extend(list(set(patch_df[patch_df['pid']==p]['surv'])))
            status.extend(list(set(patch_df[patch_df['pid']==p]['status'])))
            weights = [sum(patch_df[patch_df['pid']==p]['cluster']==c)
                           for c in selected_clusters]
            total_valid_patches = sum(weights)
            if total_valid_patches:
                weights = [float(w) / total_valid_patches for w in weights]
                fea_tmp = []
                for c in range(len(selected_clusters)):
                    # can also try DataFrame.assign() to do this neatly
                    if weights[c]:
                        # fea_c = weights[c] * patch_df[(patch_df['pid']==p) & (patch_df['cluster']==selected_clusters[c])].loc[:,'fea_0':'fea_%d'%(fea_dim-1)].mean()
                        fea_c = weights[c] * patch_df[(patch_df['pid']==p) & (patch_df['cluster']==selected_clusters[c])].iloc[:,5:].mean()
                        fea_tmp.extend(fea_c)
                    else:
                        fea_tmp.extend([0.0] * fea_dim)
            else:
                fea_tmp = [0.0] * len(selected_clusters) * fea_dim
            features.append(fea_tmp)

        features = pd.DataFrame(features)
        patient_feas = pd.DataFrame(zip(pid, surv, status))
        patient_feas.columns = ['pid','surv', 'status']
        patient_feas = patient_feas.join(features)
        fea_column_names = ['fea_%d'%i for i in range(features.shape[1])]
        patient_feas.columns = ['pid','surv', 'status'] + fea_column_names

        return patient_feas