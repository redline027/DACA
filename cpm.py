import numpy as np

def compute_performance_measures(positions, ground_truth, distance_precision_threshold, pascal_threshold, dirname):

    # [distance_precision, PASCAL_precision, average_center_location_error] = ...
    #    compute_performance_measures(positions, ground_truth, distance_precision_threshold, PASCAL_threshold)
    #
    # For the given tracker output positions and ground truth it computes the:
    # * Distance Precision at the specified threshold (20 pixels as default if
    # omitted)
    # * PASCAL Precision at the specified threshold (0.5 as default if omitted)
    # * Average Center Location error (CLE).
    #
    # The tracker positions and ground truth must be Nx4-matrices where N is
    # the number of time steps in the tracking. Each row has to be on the form
    # [c1, c2, s1, s2] where (c1, c2) is the center coordinate and s1 and s2 
    # are the size in the first and second dimension respectively (the order of 
    # x and y does not matter here).

    if distance_precision_threshold == None:
        distance_precision_threshold = 20

    if pascal_threshold == None:
        pascal_threshold = 0.5

    if positions.shape[0] != ground_truth.shape[0]:
        print('Could not calculate precisions, because the number of ground')
        print('truth frames does not match the number of tracked frames.')
        return

    #calculate distances to ground truth over all frames
    a = positions[:,0] - ground_truth[:,0]
    a = np.power(a, 2)
    b = positions[:,1] - ground_truth[:,1]
    b = np.power(b, 2)
    distances = np.sqrt(a + b)
    distances[np.isnan(distances)] = []

    #calculate distance precision
    distance_precision = np.count_nonzero(distances < distance_precision_threshold) / distances.size

    #calculate average center location error (CLE)
    average_center_location_error = np.mean(distances)

    #calculate the overlap in each dimension
    a = positions[:,0] + positions[:,2] / 2
    b = ground_truth[:,0] + ground_truth[:,2] / 2
    c = positions[:,0] - positions[:,2] / 2
    d = ground_truth[:,0] - ground_truth[:,2] / 2
    overlap_height = np.empty(a.size)
    for i in range(a.size):
        overlap_height[i] = min(a[i], b[i]) - max(c[i], d[i])
    a = positions[:,1] + positions[:,3] / 2
    b = ground_truth[:,1] + ground_truth[:,3] / 2
    c = positions[:,1] - positions[:,3] / 2
    d = ground_truth[:,1] - ground_truth[:,3] / 2
    overlap_width = np.empty(a.size)
    for i in range(a.size):
        overlap_width[i] = min(a[i], b[i]) - max(c[i], d[i])

    # if no overlap, set to zero
    overlap_height[overlap_height < 0] = 0
    overlap_width[overlap_width < 0] = 0

    # remove NaN values (should not exist any)
    valid_ind = ~np.isnan(overlap_height) & ~np.isnan(overlap_width)
    valid_ind = valid_ind.astype('int')

    # calculate area
    size = np.count_nonzero(valid_ind)
    overlap_area = np.empty(size)
    tracked_area = np.empty(size)
    ground_truth_area = np.empty(size)
    j = 0
    for i in range(positions.shape[0]):
        if valid_ind[i]:
            overlap_area[j] = overlap_height[i] * overlap_width[i]
            tracked_area[j] = positions[i, 2] * positions[i, 3]
            ground_truth_area[j] = ground_truth[i, 2] * ground_truth[i, 3]
            j += 1

    # calculate PASCAL overlaps
    overlaps = overlap_area / (tracked_area + ground_truth_area - overlap_area)

    # calculate PASCAL precision
    pascal_precision = np.count_nonzero(overlaps >= pascal_threshold) / overlaps.size

    f_out = open(dirname + '/distances.txt', 'w')
    for i in range(distances.size):
        f_out.write(str(distances[i]) + '\n')

    f_out.close()

    f_out = open(dirname + '/overlaps.txt', 'w')
    for i in range(overlaps.size):
        f_out.write(str(overlaps[i]) + '\n')

    f_out.close()

    return distance_precision, pascal_precision, average_center_location_error
