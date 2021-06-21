import pdb
import os
import shutil
import time
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from PIL import Image
import cv2
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
import glob
import re
from collections import defaultdict
import sys
import random

# gpu clustering from: https://github.com/facebookresearch/deepcluster/blob/master/clustering.py
#import faiss
import mkl
mkl.get_max_threads()


def get_file_list(fPath,
                  fType,
                  str_include='',
                  str_exclude=' ',
                  exclude_dirs=None):
    """Get the file list in the path `fPath` according to the given file type `fType`
    """
    if not isinstance(fType, str):
        print('File type should be string!')
        return
    else:
        all_imgs = []
        if exclude_dirs != None:
            for root, _, fl in tqdm(list(os.walk(fPath)),
                                    dynamic_ncols=True,
                                    ascii=False,
                                    desc='Indexing data dir:{}'.format(fPath)):
                for f in fl:
                    if (f.endswith(fType) and str_include in f and
                            str_exclude not in f and
                            f[:60] not in exclude_dirs):
                        all_imgs.append(os.path.join(root, f))
                        # import pdb
                        # pdb.set_trace()
            return all_imgs

        else:
            return [
                os.path.join(root, f)
                for root, _, fl in list(os.walk(fPath))
                for f in fl
                if (f.endswith(fType) and str_include in f and
                    str_exclude not in f)
            ]


def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(42)

    clus.niter = 300  # 训练迭代次数
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]


def get_pca_reducer_incremental(tr_tensor, n_comp=10):
    # Apply Incremental PCA on the training images
    bs = 100
    pca = IncrementalPCA(n_components=n_comp, batch_size=bs)

    for i in range(0, len(tr_tensor), bs):
        print(f"fitting {i//bs} th batch")
        pca.partial_fit(tr_tensor[i:i+bs, :])

    return pca


def combine_images_into_tensor(img_fnames, size=512):
    """
    Given a list of image filenames, read the images, flatten them
    and return a tensor such that each row contains one image.
    Size of individual image: 320*320
    """
    # Initialize the tensor
    # tensor = np.zeros((len(img_fnames), size * size*3))
    #
    # for i, fname in enumerate(img_fnames):
    #     img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    #     tensor[i] = img.reshape(-1)

    all_feat = [] #image
    for fn in tqdm(img_fnames):
        #tmp = np.load(fn, allow_pickle=True).item()[which_feat]
        tmp = np.asarray(Image.open(fn).resize((50,50))).reshape(-1)
        #tmp = np.squeeze(tmp)  # (2048,1,1)-> (2048,)
        # pdb.set_trace()
        all_feat.append(tmp)

    return np.array(all_feat)


def cluster_images(all_img_fnames, num_clusters, wsi_feat_dir, num_file):
    # Select images at random for PCA
    random.shuffle(all_img_fnames)
    tr_img_fnames = all_img_fnames#[:400]

    # Flatten and combine the images
    tr_tensor = combine_images_into_tensor(tr_img_fnames)

    # Perform PCA
    print("Learning PCA...")
    n_comp = 50
    pca = get_pca_reducer_incremental(tr_tensor, n_comp)

    # Transform images in batches
    print("applying PCA transformation")
    points = np.zeros((len(all_img_fnames), n_comp))
    batch_size = 50
    for i in range(0, len(all_img_fnames), batch_size):
        print(f"Transforming {i//25} th batch")
        batch_fnames = all_img_fnames[i:i+batch_size]
        all_tensor = combine_images_into_tensor(batch_fnames)
        points[i:i+batch_size] = pca.transform(all_tensor)

    # Cluster
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(points)

    # Organize image filenames based on the obtained clusters
    cluster_fnames = defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        cluster_fnames[label].append(all_img_fnames[i])

        fn = all_img_fnames[i]
        write_item = (",").join([fn, fn.split('/')[-2], str(label)])
        with open(os.path.join( '/home/cy/ssd_4t/nlst/nlst_patch_cluster',
        wsi_feat_dir.split('/')[-2]+'_'+str(num_file)+'_cls'+str(num_clusters)
        +'.csv'),'a') as cluster_file:
            cluster_file.write(write_item + '\n')

    return cluster_fnames


def cluster_wsis(wsi_feat_dir, pca, num_clusters, num_file = 1000, load_all = False):
    start = time.time()
    if load_all:
        all_file = get_file_list(wsi_feat_dir,
                                 'jpg',
                                 str_include='',
                                 str_exclude=' ')
    else:
        all_file = []
        all_subfolder = glob.glob(wsi_feat_dir+'/*')

        # all_cancer_dir = os.listdir(wsi_feat_dir)
        # print('files num ', len(all_cancer_dir))
        # all_subfolder = []
        # for each_cancer in all_cancer_dir:
        #     cancer_dir = os.path.join(wsi_feat_dir, each_cancer)
        #     print(cancer_dir)
        #     if not os.path.isdir(cancer_dir):
        #         continue
        #     wsi_dirs = glob.glob(cancer_dir+'/*')
        #     print(len(wsi_dirs))
        #     all_subfolder.extend(wsi_dirs)

        random.seed(4)
        # random.shuffle(all_subfolder)
        #all_subfolder = all_subfolder[50:]
        print(all_subfolder[:5])

        for each_subfolder in tqdm(all_subfolder,desc = 'loading file list'):
            each_fl = get_file_list(each_subfolder,
                                     'jpg',
                                     str_include='',
                                     str_exclude=' ')
            random.shuffle(each_fl)
            all_file += each_fl[:num_file]
        print('total files: {}'.format(len(all_file)))

    clustered_fnames = cluster_images(all_file, num_clusters, wsi_feat_dir, num_file)

    return clustered_fnames


if __name__ == "__main__":
    data_dir = '/home/cy/ssd_4t/nlst/nlst_patch'
    #data_dir = '/home/cy/ssd_4t/Chest_anhui/anhui_patch_xlz'
    #data_dir = '/home/cy/ssd_4t/stainGan/cluster'
    #which_feat = 'global_avgpool'

    wsi_dirs = os.listdir(data_dir)
    print(len(wsi_dirs))

    # all_cancer_dir = os.listdir(data_dir)
    # for each_cancer in all_cancer_dir:
    #     cancer_dir = os.path.join(data_dir, each_cancer)
    #     wsi_dirs = os.listdir(cancer_dir)
    #
    #     pca = PCA(n_components=512)
    #     for case_name in tqdm(wsi_dirs):
    #         wsi_abs_path = os.path.join(cancer_dir, case_name)
    #         cluster_wsi(wsi_abs_path, which_feat, pca)


    # cluster all patches
    pca = PCA(n_components=512)
    num_clusters = 10
    cluster_wsis(data_dir, pca, num_clusters, load_all = True)

    
