import numpy as np
import math
import get_feature_map

def get_subwindow(im, pos, sz, non_pca_features, pca_features, w2c):

    # [out_npca, out_pca] = get_subwindow(im, pos, sz, non_pca_features, pca_features, w2c)
    #
    # Extracts the non-PCA and PCA features from image im at position pos and
    # window size sz. The features are given in non_pca_features and
    # pca_features. out_npca is the window of non-PCA features and out_pca is
    # the PCA-features reshaped to [prod(sz) num_pca_feature_dim]. w2c is the
    # Color Names matrix if used.

    if np.isscalar(sz):  #square sub-window
        sz = [sz, sz]

    xs = np.arange(1, sz[1] + 1)
    ys = np.arange(1, sz[0] + 1)
    xs = math.floor(pos[1]) + xs - math.floor(sz[1]/2)
    ys = math.floor(pos[0]) + ys - math.floor(sz[0]/2)

    #check for out-of-bounds coordinates, and set them to the values at
    #the borders
    xs[xs < 1] = 1
    ys[ys < 1] = 1
    xs[xs > im.shape[1]] = im.shape[1]
    ys[ys > im.shape[0]] = im.shape[0]

    #extract image
    im_patch = im[ys, xs, :]

    # compute non-pca feature map
    if non_pca_features.size != 0:
        out_npca = get_feature_map.get_feature_map(im_patch, non_pca_features, w2c)
    else:
        out_npca = []

    # compute pca feature map
    if pca_features.size != 0:
        temp_pca = get_feature_map.get_feature_map(im_patch, pca_features, w2c)
        out_pca = temp_pca.reshape((np.prod(sz), temp_pca.shape[2]))
    else:
        out_pca = []

    return out_npca, out_pca
