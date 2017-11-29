import numpy as np
import h5py
import math
import time
import feature_projection

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
    pos = params.init_pos
    target_sz = params.wsize

    visualization = params.visualization

    num_frames = len(img_files)

    arrays = {}
    f = h5py.File('w2crs.mat')
    for k, v in f.items():
        arrays[k] = np.array(v)
    w2c = arrays['w2crs']

    use_dimensionality_reduction = 0
    if len(compressed_features) != 0:
        use_dimensionality_reduction = 1

    # window size, taking padding into account
    sz = np.empty(2)
    for i in range(2):
        sz[i] = math.floor(target_sz[i] * (1 + padding));

    # desired output (gaussian shaped), bandwidth proportional to target size
    output_sigma = math.sqrt(np.prod(target_sz)) * output_sigma_factor;
    # TODO: research how it works
    #rs, cs = np.meshgrid([1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2]);
    #x.^2 = np.power(x,2)
    #y = exp(-0.5 / output_sigma^2 * (rs.^2 + cs.^2));
    #fft2(y): np.fft.fft2(y)
    #single(a): a = a.astype('float32')
    #yf = single(fft2(y));

    # store pre-computed cosine window
    cos_window = np.dot(np.hanning(sz[0]), np.transpose(np.hanning(sz[1])))
    cos_window = cos_window.astype('float32')

    # to calculate precision
    positions = np.zeros(len(img_files), 4)

    # initialize the projection matrix
    projection_matrix = []

    # to calculate fps
    time = 0

    #???
    z_npca = []
    z_pca = []

    for frame in range(2): #num_frames instead 1
        # load image
        # im = imread([video_path img_files{frame}]);

        t = time.time()

        if frame > 0
            # compute the compressed learnt appearance
            zp = feature_projection.feature_projection(z_npca, z_pca, projection_matrix, cos_window);

            # extract the feature map of the local image patch
            #[xo_npca, xo_pca] = get_subwindow(im, pos, sz, non_compressed_features, compressed_features, w2c);

            # do the dimensionality reduction and windowing
            #x = feature_projection(xo_npca, xo_pca, projection_matrix, cos_window);

            # calculate the response of the classifier
            #kf = fft2(dense_gauss_kernel(sigma, x, zp));
            #response = real(ifft2(alphaf_num .* kf ./ alphaf_den));

            # target location is at the maximum response
            #[row, col] = find(response == max(response(:)), 1);
            #pos = pos - floor(sz/2) + [row, col];

    positions = {}
    fps = {}
    return positions, fps
