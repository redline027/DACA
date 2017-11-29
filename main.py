import numpy as np
import choose_video
import load_video_info
import color_tracker
import math

base_path = 'sequences/'

#parameters according to the paper
class Params:
    # extra area surrounding the target
    padding = 1.0
    # spatial bandwidth (proportional to target)
    output_sigma_factor = 1.0/16
    # gaussian kernel bandwidth
    sigma = 0.2
    # regularization (denoted "lambda" in the paper)
    lmbda = 1e-2
    # learning rate for appearance model update scheme (denoted "gamma" in the paper)
    learning_rate = 0.075
    # learning rate for the adaptive dimensionality reduction (denoted "mu" in the paper)
    compression_learning_rate = 0.15
    # features that are not compressed, a cell  with strings (possible choices: 'gray', 'cn')
    non_compressed_features = ['gray']
    # features that are compressed, a cell with strings (possible choices: 'gray', 'cn')
    compressed_features = ['cn']
    # the dimensionality of the compressed features
    num_compressed_dim = 2
    visualization = 1

params = Params()

video = choose_video.choose_video(base_path)
if video == '':
     sys.exit() #user cancelled

img_files, pos, target_sz, ground_truth, video_path = load_video_info.load_video_info(base_path, video)

params.init_pos = np.zeros(2)
for i in range(2):
    params.init_pos[i] = math.floor(pos[i]) + math.floor(target_sz[i]/2)
params.wsize = np.zeros(2)
for i in range(2):
    params.wsize[i] = math.floor(target_sz[i])
params.img_files = img_files
params.video_path = video_path

positions, fps = color_tracker.color_tracker(params)

print('Hi')
