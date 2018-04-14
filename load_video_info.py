import numpy as np
import tkinter
from tkinter import filedialog

def load_video_info(base_path, video):

    # [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(video_path)

    root = tkinter.Tk()
    root.withdraw()
    dirname = filedialog.askdirectory(parent=root, title='Please select a directory')

    deer = False

    try:
        f_gt = open(dirname + '/groundtruth.txt', 'r')
        f_out = open(dirname + '/gt.txt', 'w')

        cnt = 0

        for line in f_gt:
            coods = line.split(',')
            x1 = float(coods[0])
            y1 = float(coods[1])
            x2 = float(coods[2])
            y2 = float(coods[3])
            x3 = float(coods[4])
            y3 = float(coods[5])
            x4 = float(coods[6])
            y4 = float(coods[7])
            x_min = min(x1, x2, x3, x4)
            y_min = min(y1, y2, y3, y4)
            x_max = max(x1, x2, x3, x4)
            y_max = max(y1, y2, y3, y4)
            hight = int(round(y_max - y_min))
            width = int(round(x_max - x_min))
            f_out.write(str(x_min) + ',' + str(y_min) + ',' + str(width) + ',' + str(hight) + '\n')
            cnt += 1

        f_gt.close()
        f_out.close()

        f_out = open(dirname + '/frames.txt', 'w')
        f_out.write(str('1,' + str(cnt)))
        f_out.close()

    except Exception:
        deer = True

    video_path = dirname

    if deer:
        frames_file = video_path + '/deer_frames.txt'
    else:
        frames_file = video_path + '/frames.txt'
    #TODO: check file !!!!!

    f = open(frames_file, 'r')
    frames = f.read().split(',')
    for i in range(len(frames)):
        frames[i] = int(frames[i])
    f.close()

    if deer:
        gt_file = video_path + '/deer_gt.txt'
    else:
        gt_file = video_path + '/gt.txt'
    #TODO: check file !!!!!

    ground_truth_list = []
    f = open(gt_file, 'r')
    for line in f:
        line_list = line.split(',')
        for i in range(len(line_list)):
            line_list[i] = float(line_list[i])
        ground_truth_list.append(line_list)
    f.close()

    ground_truth = np.array(ground_truth_list)

    #set initial position and size
    target_sz = [ground_truth[0,3], ground_truth[0,2]]
    pos = [ground_truth[0,1], ground_truth[0,0]]

    a = ground_truth[:, [1, 0]]
    b = (ground_truth[:,[3,2]] - 1) / 2
    ground_truth = np.concatenate((a + b, ground_truth[:,[3,2]]), axis = 1)

    # TODO: check; not only jpg;
    img_files = []
    if deer:
        for i in range(frames[0], frames[1] + 1):
            img_files.append('img%05d.jpg' % i)
    else:
        for i in range(frames[0], frames[1] + 1):
            img_files.append('%08d.jpg' % i)

    return img_files, pos, target_sz, ground_truth, video_path
