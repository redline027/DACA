import numpy as np
import cv2
import h5py
import im2c


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_feature_map(im_patch, features, w2c):

    # out = get_feature_map(im_patch, features, w2c)
    #
    # Extracts the given features from the image patch. w2c is the
    # Color Names matrix, if used.

    #if nargin < 3:
    #    w2c = []

    # the names of the features that can be used
    valid_features = ['gray', 'cn']

    # the dimension of the valid features
    # !!! WAS feature_levels = [1 10]'; !!!!!!!
    feature_levels = np.array([1, 10])[np.newaxis]
    feature_levels = feature_levels.T

    num_valid_features = len(valid_features)
    used_features = np.zeros((num_valid_features, 1), dtype=bool)

    # get the used features
    for i in range(num_valid_features):
        cmprd = np.zeros((len(features)), dtype=bool)
        for j in range(len(features)):
            cmprd[j] = (valid_features[i].lower() == features[j].lower())
        used_features[i] = any(cmprd)

    # total number of used feature levels
    num_feature_levels = sum(feature_levels * used_features)
    num_feature_levels = num_feature_levels[0]

    level = 0

    # If grayscale image
    if len(im_patch.shape) == 2:
        # Features that are available for grayscale sequances
        
        # Grayscale values (image intensity)
        out = im_patch.astype(float) / 255 - 0.5
    else:
        # Features that are available for color sequances
        
        # allocate space (for speed)
        out = np.zeros((im_patch.shape[0], im_patch.shape[1], num_feature_levels), dtype=float)
        
        # Grayscale values (image intensity)
        if used_features[0]:
            im_patch_gray = rgb2gray(im_patch)
            for i in range(im_patch_gray.shape[0]):
                for j in range(im_patch_gray.shape[1]):
                    im_patch_gray[i][j] = round(im_patch_gray[i][j])

            out[:,:,level] = im_patch_gray.astype(float) / 255 - 0.5;
            level = level + feature_levels[0][0]

        # Color Names
        if used_features[1]:
            # !!! TODO: need diferent check
            if w2c.shape == 0:
                # load the RGB to color name matrix if not in input
                arrays = {}
                f = h5py.File('w2crs.mat')
                for k, v in f.items():
                    arrays[k] = np.array(v)
                w2c = arrays['w2crs']
                w2c = w2c.T

            # extract color descriptor
            out_tmp = im2c.im2c(im_patch.astype(float), w2c, -2)

            level_tmp = level + np.arange(10)
            out[:, :, level_tmp] = out_tmp

            level = level + feature_levels[1]

    return out