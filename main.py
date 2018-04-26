import numpy as np
import choose_video
import load_video_info
import color_tracker
import cpm
import math

base_path = 'sequences/'

#parameters according to the paper
class Params:
    # extra area surrounding the target
    padding = 1
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
     exit() #user cancelled

img_files, pos, target_sz, ground_truth, video_path, dirname = load_video_info.load_video_info(base_path, video)

f_out = open(dirname + '/params.txt', 'w')
f_out.write('padding ' + str(params.padding) + '\n')
f_out.write('output_sigma_factor ' + str(params.output_sigma_factor) + '\n')
f_out.write('sigma ' + str(params.sigma) + '\n')
f_out.write('lmbda ' + str(params.lmbda) + '\n')
f_out.write('learning_rate ' + str(params.learning_rate) + '\n')
f_out.write('compression_learning_rate ' + str(params.compression_learning_rate) + '\n')
f_out.write('non_compressed_features ')
for i in range(len(params.non_compressed_features)):
    f_out.write(params.non_compressed_features[i])
f_out.write('\n')
f_out.write('compressed_features ')
for i in range(len(params.compressed_features)):
    f_out.write(params.compressed_features[i])
f_out.write('\n')
f_out.write('num_compressed_dim ' + str(params.num_compressed_dim) + '\n')
f_out.write('visualization ' + str(params.visualization) + '\n')
f_out.close()

params.init_pos = np.zeros(2)
for i in range(2):
    params.init_pos[i] = math.floor(pos[i]) + math.floor(target_sz[i]/2)
params.wsize = np.zeros(2)
for i in range(2):
    params.wsize[i] = math.floor(target_sz[i])
params.img_files = img_files
params.video_path = video_path

positions, fps = color_tracker.color_tracker(params)

distance_precision, pascal_precision, average_center_location_error = cpm.compute_performance_measures(positions, ground_truth, None, None, dirname)

print('Center Location Error: %.3g pixels\nDistance Precision: %.3g %%\nOverlap Precision: %.3g %%\nSpeed: %.3g fps\n' % (average_center_location_error, 100*distance_precision, 100*pascal_precision, fps))