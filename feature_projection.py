import numpy as np

def feature_projection(x_npca, x_pca, projection_matrix, cos_window):

    # z = feature_projection(x_npca, x_pca, projection_matrix, cos_window)
    #
    # Calculates the compressed feature map by mapping the PCA features with
    # the projection matrix and concatinates this with the non-PCA features.
    # The feature map is then windowed.

    if x_pca.size == 0:
        # if no PCA-features exist, only use non-PCA
        z = x_npca
    else:
        # get dimensions
        height, width = cos_window.shape
        num_pca_in, num_pca_out = projection_matrix.shape

        # project the PCA-features using the projection matrix and reshape
        # to a window
        x_proj_pca = np.dot(x_pca, projection_matrix)
        x_proj_pca = np.reshape(x_proj_pca, (height, width, num_pca_out), order='F')

        # concatinate the feature windows
        if x_npca.size == 0:
            z = x_proj_pca
        else:
            z = np.concatenate((x_npca, x_proj_pca), axis = 2)

    # do the windowing of the output
    for i in range(z.shape[2]):
        z[:,:,i] = cos_window * z[:,:,i]

    #like in matlab
    # TODO:REMOVE
    z = np.around(z,decimals=4)

    return z
