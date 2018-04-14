import numpy as np
import h5py
import math
import cv2
import time
import feature_projection
import get_subwindow
import dense_gauss_kernel
from PIL import Image

def color_tracker(params):

    # [positions, fps] = color_tracker(params)

    # parameters
    padding = params.padding
    output_sigma_factor = params.output_sigma_factor
    sigma = params.sigma
    lmbda = params.lmbda
    learning_rate = params.learning_rate
    compression_learning_rate = params.compression_learning_rate
    non_compressed_features = params.non_compressed_features
    compressed_features = params.compressed_features
    num_compressed_dim = params.num_compressed_dim

    video_path = params.video_path
    img_files = params.img_files
    #TODO: check here floor
    pos = np.empty(2)
    target_sz = np.empty(2)
    for i in range(2):
        pos[i] = math.floor(params.init_pos[i])
        target_sz[i] = math.floor(params.wsize[i])

    visualization = params.visualization

    num_frames = len(img_files)

    arrays = {}
    f = h5py.File('w2crs.mat')
    for k, v in f.items():
        arrays[k] = np.array(v)
    w2c = arrays['w2crs']
    w2c = w2c.T

    use_dimensionality_reduction = 0
    if len(compressed_features) != 0:
        use_dimensionality_reduction = 1

    # window size, taking padding into account
    sz = np.empty(2)
    for i in range(2):
        sz[i] = math.floor(target_sz[i] * (1 + padding));

    # desired output (gaussian shaped), bandwidth proportional to target size
    output_sigma = math.sqrt(np.prod(target_sz)) * output_sigma_factor;
    a = np.arange(1, sz[0] + 1) - math.floor(sz[0]/2.0)
    b = np.arange(1, sz[1] + 1) - math.floor(sz[1]/2.0)
    rs, cs = np.meshgrid(a, b)
    rs = rs.T
    cs = cs.T
    a = np.power(rs, 2) + np.power(cs, 2)
    y = np.exp(-0.5 / output_sigma**2 * a)
    a = np.fft.fft2(y)
    yf = a.astype('complex')

    # store pre-computed cosine window
    a = np.array(np.hanning(sz[0]))[np.newaxis]
    b = np.array(np.hanning(sz[1]))[np.newaxis]
    cos_window = np.dot(a.T, b)
    cos_window = cos_window.astype('float32')

    # to calculate precision
    positions = np.zeros((len(img_files), 4))

    # initialize the projection matrix
    projection_matrix = np.zeros((0,0))
    old_cov_matrix = np.zeros((0,0))

    # to calculate fps
    time_fps = 0

    #???
    z_npca = np.zeros((0))
    z_pca = np.zeros((0))
    alphaf_num = np.zeros((0))
    alphaf_den = np.zeros((0))

    #num_frames instead 1
    for frame in range(num_frames):
        # load image
        im_open_cv = cv2.imread('%s/%s' % (video_path, img_files[frame]))
        im = np.zeros(im_open_cv.shape)

        im[:,:,0] = im_open_cv[:,:,2]
        im[:,:,1] = im_open_cv[:,:,1]
        im[:,:,2] = im_open_cv[:,:,0]

        t = time.time()

        if frame > 0:
            # compute the compressed learnt appearance
            # TODO:CHECK
            zp = feature_projection.feature_projection(z_npca, z_pca, projection_matrix, cos_window)

            # extract the feature map of the local image patch
            xo_npca, xo_pca = get_subwindow.get_subwindow(im, pos, sz, non_compressed_features, compressed_features, w2c)

            # do the dimensionality reduction and windowing
            # TODO:CHECK
            x = feature_projection.feature_projection(xo_npca, xo_pca, projection_matrix, cos_window)

            # calculate the response of the classifier
            # TODO:CHECK
            kf = dense_gauss_kernel.dense_gauss_kernel(sigma, x, zp)
            kf = np.fft.fft2(kf)
            response = np.fft.ifft2(alphaf_num * kf / alphaf_den)
            response = np.real(response)

            # target location is at the maximum response
            response_dots = response.reshape((response.shape[0] * response.shape[1], 1), order='F')
            max_resp = np.max(response_dots)
            #[row, col] = find(response == max(response(:)), 1);
            f = False
            for j in range(response.shape[1]):
                for i in range(response.shape[0]):
                    if response[i, j] == max_resp:
                        row = i
                        col = j
                        f = True
                        break
                if f:
                    break

            pos[0] = pos[0] - math.floor(sz[0]/2) + row + 1
            pos[1] = pos[1] - math.floor(sz[1]/2) + col + 1

        # extract the feature map of the local image patch to train the classifer
        xo_npca, xo_pca = get_subwindow.get_subwindow(im, pos, sz, non_compressed_features, compressed_features, w2c)

        if frame == 0:
            # initialize the appearance
            z_npca = xo_npca
            z_pca = xo_pca

            # set number of compressed dimensions to maximum if too many
            num_compressed_dim = min(num_compressed_dim, xo_pca.shape[1])
        else:
            # update the appearance
            # TODO:CHECK
            z_npca = (1 - learning_rate) * z_npca + learning_rate * xo_npca
            z_pca = (1 - learning_rate) * z_pca + learning_rate * xo_pca

        # if dimensionality reduction is used: update the projection matrix
        if use_dimensionality_reduction:
            # compute the mean appearance
            data_mean = np.mean(z_pca, axis=0)

            # substract the mean from the appearance to get the data matrix
            data_matrix = np.subtract(z_pca, data_mean)

            # calculate the covariance matrix
            cov_matrix = 1 / (np.prod(sz) - 1) * np.dot(data_matrix.T, data_matrix)

            # calculate the principal components (pca_basis) and corresponding variances
            if frame == 0:
                pca_basis, pca_variances, V = np.linalg.svd(cov_matrix)
                pca_variances = np.diag(pca_variances)
            else:
                #TODO:CHECK
                tmp = (1 - compression_learning_rate) * old_cov_matrix + compression_learning_rate * cov_matrix
                pca_basis, pca_variances, V = np.linalg.svd(tmp)
                pca_variances = np.diag(pca_variances)

            # calculate the projection matrix as the first principal
            # components and extract their corresponding variances
            projection_matrix = pca_basis[:, 0:num_compressed_dim]
            projection_variances = pca_variances[0:num_compressed_dim, 0:num_compressed_dim]

            if frame == 0:
                # initialize the old covariance matrix using the computed
                # projection matrix and variances
                tmp = np.dot(projection_matrix, projection_variances)
                old_cov_matrix = np.dot(tmp, projection_matrix.T)
            else:
                # update the old covariance matrix using the computed
                # projection matrix and variances
                #TODO:CHECK
                tmp = np.dot(projection_matrix, projection_variances)
                tmp = np.dot(tmp, projection_matrix.T)
                old_cov_matrix = (1 - compression_learning_rate) * old_cov_matrix + compression_learning_rate * tmp

            # project the features of the new appearance example using the new
            # projection matrix
            x = feature_projection.feature_projection(xo_npca, xo_pca, projection_matrix, cos_window)
 
            # calculate the new classifier coefficients
            tmp = dense_gauss_kernel.dense_gauss_kernel_2(sigma, x)
            kf = np.fft.fft2(tmp)
            new_alphaf_num = yf * kf
            new_alphaf_den = kf * (kf + lmbda)

            if frame == 0:
                # first frame, train with a single image
                alphaf_num = new_alphaf_num
                alphaf_den = new_alphaf_den
            else:
                # subsequent frames, update the model
                #TODO:CHECK
                alphaf_num = (1 - learning_rate) * alphaf_num + learning_rate * new_alphaf_num
                alphaf_den = (1 - learning_rate) * alphaf_den + learning_rate * new_alphaf_den

            #save position
            positions[frame,0] = pos[0]
            positions[frame,1] = pos[1]
            positions[frame,2] = target_sz[0]
            positions[frame,3] = target_sz[1]

            time_fps = time_fps + (time.time() - t)

            #visualization
            if visualization == 1:
                rect = np.zeros(4)
                rect[0] = pos[1] - target_sz[1]/2
                rect[1] = pos[0] - target_sz[0]/2
                rect[2] = target_sz[1]
                rect[3] = target_sz[0]
                rect = rect.astype('int')

                #if frame == 0:  #first frame, create GUI
                #    text_handle = text(10, 10, int2str(frame));
                #    set(text_handle, 'color', [0 1 1]);
                #else:
                    #subsequent frames, update GUI
                    #set(im_handle, 'CData', im)
                    #set(rect_handle, 'Position', rect_position)
                    #set(text_handle, 'string', int2str(frame));

                img = im_open_cv
                cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (200, 0, 0), 2)
                #cv2.destroyAllWindows()
                cv2.imshow('Frames', im_open_cv)
                cv2.waitKey(1)

    fps = num_frames / time_fps

    return positions, fps
