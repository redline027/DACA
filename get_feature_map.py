import numpy as np
import cv2
import h5py

def get_feature_map(im_patch, features, w2c):

    # out = get_feature_map(im_patch, features, w2c)
    #
    # Extracts the given features from the image patch. w2c is the
    # Color Names matrix, if used.

    if nargin < 3:
        w2c = []

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
            cmprd[i] = (valid_features[i].lower() == features[i].lower())
        used_features[i] = any(cmprd)


    # total number of used feature levels
    num_feature_levels = sum(feature_levels * used_features)

    level = 0

    # If grayscale image
    if im_patch.shape[2] == 1:
        # Features that are available for grayscale sequances
        
        # Grayscale values (image intensity)
        out = im_patch.astype(float) / 255 - 0.5
    else:
        # Features that are available for color sequances
        
        # allocate space (for speed)
        out = np.zeros((im_patch.shape[0], im_patch.shape[1], num_feature_levels), dtype=float)
        
        # Grayscale values (image intensity)
        if used_features[0]:
            cv2.cvtColor(im_patch, im_patch_gray, cv2.COLOR_RGB2GRAY)
            #level+1???? !!!!!!!!!!!
            out[:,:,level] = im_patch_gray.astype(float) / 255 - 0.5;
            level = level + feature_levels[0]
        
        # Color Names
        if used_features[1]:
            if w2c.size == 0:
                # load the RGB to color name matrix if not in input
                arrays = {}
                f = h5py.File('w2crs.mat')
                for k, v in f.items():
                    arrays[k] = np.array(v)
                w2c = arrays['w2crs']
            
            # extract color descriptor
            #out(:,:,level+(1:10)) = im2c(single(im_patch), w2c, -2);
            #level = level + feature_levels(2);

    return out