import numpy as np

def load_video_info(base_path, video):

    # [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(video_path)

    video_path = base_path + '/' + video + '/'

    frames_file = video_path + video + '_frames.txt'
    #TODO: check file !!!!!

    f = open(frames_file, 'r')
    frames = f.read().split(',')
    for i in range(len(frames)):
        frames[i] = int(frames[i])
    f.close()

    gt_file = video_path + video + '_gt.txt'
    #TODO: check file !!!!!

    ground_truth_list = []
    f = open(gt_file, 'r')
    for line in f:
        line_list = line.split(',')
        for i in range(len(line_list)):
            line_list[i] = float(line_list[i])
        ground_truth_list.append(line_list)
    f.close();

    ground_truth = np.array(ground_truth_list)

    #set initial position and size
    target_sz = [ground_truth[0,3], ground_truth[0,2]]
    pos = [ground_truth[0,1], ground_truth[0,0]]

    #TODO: ground_truth - magic
    #ground_truth = [ground_truth(:,[2,1]) + (ground_truth(:,[4,3]) - 1) / 2 , ground_truth(:,[4,3])];

    video_path = video_path + 'imgs/'
    # TODO: check; not only jpg;
    img_files = []
    for i in range(frames[0], frames[1] + 1):
        img_files.append('img%05d.jpg' % i)

    return img_files, pos, target_sz, ground_truth, video_path
