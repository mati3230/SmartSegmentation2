import numpy as np
import open3d as o3d
import os


def mkdir(directory):
    """Method to create a new directory.

    Parameters
    ----------
    directory : str
        Relative or absolute path.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


def file_exists(filepath):
    """Check if a file exists.

    Parameters
    ----------
    filepath : str
        Relative or absolute path to a file.

    Returns
    -------
    boolean
        True if the file exists.

    """
    return os.path.isfile(filepath)


def create_segments(arr, axis=None):
    """Creates a sorted segment vector for each unique element of the input
    array.

    Parameters
    ----------
    arr : np.ndarray
        Unsorted Segment vector.
    axis : int
        Axis along to which the unique elements should be created

    Returns
    -------
    tuple(np.ndarray, int, np.ndarray)
        Returns a sorted segment values, number of segments, unique segment values.

    """
    uni_arr, uni_idxs, counts = np.unique(
        arr, return_index=True, return_counts=True, axis=axis)
    segment = 0
    segments = np.zeros((arr.shape[0], ), dtype=np.int32)
    for i in range(uni_arr.shape[0]):
        segment += 1
        idx = uni_idxs[i]
        count = counts[i]
        segments[idx:idx+count] = segment
    return segments, segment, uni_arr


def iter_uni_elements(arr, uni_arr, idxs, counts, iter_method):
    """Helper method to apply custom operations to an array.

    Parameters
    ----------
    arr : np.ndarray
        Sorted array.
    uni_arr : np.ndarray
        Unique values of the input array.
    idxs : np.ndarray
        Indices of the unique values.
    counts : np.ndarray
        Counts of the unique values.
    iter_method : function
        Function that accepts the input array (np.ndarray), a unique element
        value (int), a start index of the unique element value (int), a count
        value for number of the unique element vlaues in the input array (int).
    """
    for i in range(uni_arr.shape[0]):
        uni_elem = uni_arr[i]
        idx = idxs[i]
        count = counts[i]
        iter_method(arr, uni_elem, idx, count)


def is_arr_visited(lst, x):
    """Checks if the array x is in the list lst.

    Parameters
    ----------
    lst : list(np.ndarray)
        List of arrays.
    x : np.ndarray
        Array that should be checked.

    Returns
    -------
    boolean
        True if x is in the list.

    """
    for arr in lst:
        if np.array_equal(x, arr):
            return True
    return False


def argsort_2D_mat_by_vec(X):
    """Sorts a matrix by its row vectors.

    Parameters
    ----------
    X : np.ndarray
        2D Matrix.

    Returns
    -------
    np.ndarray
        Sorted matrix.

    """
    visited = []
    offset = 0

    result = np.zeros((X.shape[0], ), dtype=np.int32)
    for i in range(X.shape[0]):
        x_i = X[i]
        # if x_i in visited:
        if is_arr_visited(visited, x_i):
            continue

        n = 0
        visited.append(x_i)
        result[offset + n] = i
        n += 1

        for j in range(i+1, X.shape[0]):
            x_j = X[j]
            # if x_i != x_j:
            if not np.array_equal(x_i, x_j):
                continue
            result[offset + n] = j
            n += 1
        offset += n

    return result


def filter_small_segments(arr, small_segment_size, axis=None, verbose=False):
    """Short summary.

    Parameters
    ----------
    arr : np.ndarray
        Segments as sorted array with integer values.
    small_segment_size : int
        Threshold to classify a small segment.
    axis : int
        Specific axis of input array where the small segments should be
        filtered.
    verbose : boolean
        If True, the remaining number of segments will be printed.

    Returns
    -------
    np.ndarray
        Indexes of segments that are smaller than small_segment_size.

    """
    uni_arr, uni_idxs, counts = np.unique(
        arr, return_index=True, return_counts=True, axis=axis)

    small_segments_lst = []

    def _filter_small_segments(arr, uni_elem, idx, count):
        if count < small_segment_size:
            small_segments_lst.append((idx, count))
    iter_uni_elements(
        arr=arr,
        uni_arr=uni_arr,
        idxs=uni_idxs,
        counts=counts,
        iter_method=_filter_small_segments)
    if verbose:
        print("n remaining segments:", uni_arr.shape[0] - len(small_segments_lst))

    # create idxs of small segments and add idx_to_del
    idx_to_del = np.zeros((0, 1), dtype=np.int32)
    for i in range(len(small_segments_lst)):
        idx, count = small_segments_lst[i]
        r = np.arange(start=idx, stop=idx + count, dtype=np.int32)
        idx_to_del = np.vstack((idx_to_del, r[:, None]))

    return idx_to_del


def get_base_colors():
    """Generate some colors.

    Returns
    -------
    np.ndarray
        2D Matrix where each row vector represents a color.
    """
    return np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]])


def generate_colors(max_colors):
    """Generate variations of the base colors that are defined in the function
    get_base_colors().

    Parameters
    ----------
    max_colors : int
        Number of colors that are needed.

    Returns
    -------
    np.ndarray
        2D Matrix where each row vector represents a color.

    """
    colors = get_base_colors()
    tmp_colors = np.array(colors, copy=True)
    n_colors = colors.shape[0]

    max_col = int(np.ceil(max_colors / n_colors))
    for i in range(1, max_col):
        tmp_colors = np.vstack((tmp_colors, (i/max_col) * colors))
    return tmp_colors


def coordinate_system():
    """Returns a coordinate system.

    Returns
    -------
    o3d.geometry.LineSet
        The lines of a coordinate system that are colored in red, green
        and blue.

    """
    line_set = o3d.geometry.LineSet()
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    lines = np.array([[0, 1], [0, 2], [0, 3]]).astype(int)
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


def render_point_cloud(
        P, segments=None, colors=None, animate=False, x_speed=2.5, y_speed=0.0):
    """Displays a point cloud.

    Parameters
    ----------
    P : np.ndarray
        Nx3 or Nx6 matrix. N is the number of points. A point should have at
        least 3 spatial coordinates and can have optionally 3 color values.
    segments : np.ndarray
        A vector that contains a segment number for each point.
    colors : np.ndarray
        A 2D matrix where each row vector represents a color. Each label of the
        segments vector will be assigned to a color.
    animate : boolean
        If True, the point cloud will be rotated with x_speed and y_speed. A
        simulation of dragging the mouse in standard rendering mode is
        simulated.
    x_speed : float
        Used if point cloud will be animated. Strength if the horizontal mouse
        drag.
    y_speed : float
        Used if point cloud will be animated. Strength if the vertical mouse
        drag.

    Returns
    -------
    int
        -1 in case of an exception and 1 otherwise.

    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P[:, :3])
    if segments is not None:
        col_mat = np.zeros((P.shape[0], 3))
        segment_values = np.unique(segments)
        n_segments = segment_values.shape[0]
        if n_segments > colors.shape[0]:
            return n_segments
        for i in range(n_segments):
            segment = segment_values[i]
            color = colors[i, :]
            idx = np.where(segments == segment)[0]
            col_mat[idx, :] = color
        pcd.colors = o3d.utility.Vector3dVector(col_mat)
    else:
        try:
            # print(P[:5, 3:6] / 255.0)
            pcd.colors = o3d.utility.Vector3dVector(P[:, 3:6] / 255.0)
        except Exception as e:
            print(e)
            return -1
    if animate:
        def rotate_view(vis):
            ctr = vis.get_view_control()
            ctr.rotate(x_speed, y_speed)
            return False
        o3d.visualization.draw_geometries_with_animation_callback(
            [pcd, coordinate_system()], rotate_view)
    else:
        o3d.visualization.draw_geometries([pcd, coordinate_system()])
    return 1


def render_point_cloud4(
        P,
        P_idxs,
        neighbour_idxs,
        P_color=np.array([1, 0, 0]),
        neighbour_color=np.array([0, 0, 1])):
    """Visualize a superpoint and a neighbour superpoint.

    Parameters
    ----------
    P : np.ndarray
        The point cloud as NxM matrix. N represents the number of points. A
        point should have at least 3 spatial coordinates.
    P_idxs : np.ndarray
        Indices of the superpoint.
    neighbour_idxs : np.ndarray
        Indices of the neighbour superpoint.
    P_color : np.ndarray
        Color which will be applied to the superpoints with P_idxs.
    neighbour_color : np.ndarray
        Color which will be applied to the superpoints with neighbour_idxs.

    Returns
    -------
    int
        1 to indicate a successful rendering process

    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P[:, :3])
    col_mat = np.zeros((P.shape[0], 3))
    col_mat[P_idxs] = P_color
    col_mat[neighbour_idxs] = neighbour_color
    pcd.colors = o3d.utility.Vector3dVector(col_mat)
    o3d.visualization.draw_geometries([pcd, coordinate_system()])
    return 1


def get_remaining_cloud(P, segment_indxs):
    """Filter points of the point cloud P by inserting the indices that should
    be filtered.

    Parameters
    ----------
    P : np.ndarray
        Point cloud P.
    segment_indxs : np.ndarray
        Indices of points that should be filtered.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        The remaining point cloud indexes by subtracting the segmented indexes.
        The remaining points.

    """
    indxs = np.arange(P.shape[0])
    indxs = np.delete(indxs, segment_indxs, axis=0)
    return indxs, P[indxs]


def segment(
        i,
        segment_nr,
        remaining_indxs,
        segment_candidates,
        segments,
        segment_indxs):
    """Applies the i-th segment candidate to a point cloud.

    Parameters
    ----------
    i : int
        Index to segment the i-th segment candidate..
    segment_nr : int
        Current segment number. The increment will be applied to the new
        segment.
    remaining_indxs : np.ndarray
        Indexes of the point cloud that are remained after a previous
        segmentation. This can be all indices of the cloud if there was no
        previous segmentation applied.
    segment_candidates : list(numpy.ndarray)
        List of segment candidates.
    segments : np.ndarray
        The assigned segment vector.
    segment_indxs : np.ndarray
        A vector of all indexes of the points that are already segmented.

    Returns
    -------
    tuple(int, np.ndarray, np.ndarray, np.ndarray)
        The incremented segment number. The remaining indices. Updated segments
        array with the incremented segment number. An array with all segmented
        indices so far.

    """
    segment_nr = segment_nr + 1
    # new segment indices of the original cloud
    new_segment_indxs = remaining_indxs[segment_candidates[i]]

    # assign segment nr to plane
    segments[new_segment_indxs] = segment_nr
    # add new segments to set of segmented indices
    segment_indxs = np.vstack((segment_indxs, new_segment_indxs[:, None]))
    return segment_nr, remaining_indxs, segments, segment_indxs


def bubble_sort(arr):
    """Applies the bubble sort algorithm.

    Parameters
    ----------
    arr : np.ndarray
        1D Array that should be sorted.

    Returns
    -------
    np.ndarray
        The sorted array.

    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j][1] < arr[j+1][1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr


def compute_error(
        assigned_segments,
        orig_segment_values,
        orig_indices,
        orig_segment_counts,
        unsegmented=0):
    """Computes the number of errorornous and unsegmented points after a
    segmentation of a point cloud. The objects will be evaluated according to
    their size. The largest object will be evaluated first.

    Parameters
    ----------
    assigned_segments : np.ndarray
        Assigned segments vector.
    orig_segment_values : np.ndarray
        The true segments vector. Should be sorted according to segment values.
    orig_indices : np.ndarray
        Description of parameter `orig_indices`.
    orig_segment_counts : np.ndarray
        Description of parameter `orig_segment_counts`.
    unsegmented : int
        Number that indicates an unsegmented point.

    Returns
    -------
    tuple(int, int, int, dict(int, int), np.ndarray, list(int))
        Number of errorornous points. Number of unsegmented points. Difference
        of the number of objects in the true and assigned segments vectors.
        Mapping from a segment of the true segments vector to a segment number
        of the assigned segments vector. Array where a reward is stored for
        each segment. List of segments that could be rewarded.

    """
    intervals = []

    _, assigned_segment_counts = np.unique(
        assigned_segments, return_counts=True)
    n_orig_segments = orig_segment_values.shape[0]
    # how much points are correct within the i-th segment?
    segment_rewards = {}
    # no more unsegmented points within segment
    finalized_segments = []
    diff = np.absolute(
        n_orig_segments - assigned_segment_counts.shape[0])
    for i in range(n_orig_segments):
        # start idx of a original segment
        idx = orig_indices[i]
        # how many original segments follow after the start idx
        count = orig_segment_counts[i]
        # np array of estimated in the range of a original segment
        obj_segments = assigned_segments[idx:idx + count]
        # which estimated segments exist in the range of a original segment and
        # how often they occur
        sorted_segments, obj_counts = np.unique(
            obj_segments, return_counts=True)

        # ( int, int, np.array, np.array )
        interval = (
            idx,
            count,
            sorted_segments,
            obj_counts,
            orig_segment_values[i])
        # sort the intervals by considering the length of a segment
        insert_idx = 0
        insert = False
        for j in range(len(intervals)):
            insert_idx = j
            # number of points in one segment
            len_points = intervals[j][1]
            if count > len_points:
                insert = True
                break
        if insert:
            intervals.insert(insert_idx, interval)
        else:
            intervals.append(interval)
    """ check sortation
    for i in range(len(intervals)-1):
        len_points = intervals[i][1]
        len_points_1 = intervals[i+1][1]
        if len_points_1 > len_points:
            raise Exception("Sortation Error: {0} > {1}".format(
                len_points_1, len_points))
    """
    n_unsegmented_points = len(np.argwhere(assigned_segments == unsegmented))
    assignments = {}
    n_errornous_points = 0
    for i in range(len(intervals)):
        # orig
        obj_idx = intervals[i][0]
        len_points = intervals[i][1]
        orig_segment = intervals[i][4]
        # estimated
        a_sorted_segments = intervals[i][2]
        a_counts = intervals[i][3]

        # estimated segments over the range of a true segment
        a_segments = assigned_segments[obj_idx:obj_idx+len_points]

        # flag that indicates if a segment is already assigned
        is_already_assigned = True
        # flag that indicates if a segment is available
        no_segment_available = False

        while is_already_assigned:
            # no more segment available for assignment
            # this happens if there are less cluster than predicted
            if a_counts.size == 0:
                no_segment_available = True
                break
            # get the index of the most frequent cluster
            j = np.argmax(a_counts)
            # get the most frequent cluster segment
            if a_sorted_segments[j] == unsegmented:
                is_already_assigned = True
            else:
                chosen_segment = a_sorted_segments[j]
                # check if segment is already assigned
                is_already_assigned = chosen_segment in assignments.values()
            if(is_already_assigned):
                # if so delete the segment and take the
                # second most segment and so forth
                a_sorted_segments = np.delete(a_sorted_segments, j)
                a_counts = np.delete(a_counts, j)
            else:
                break
        # if there are no more segments,
        # consider all the points as misclustered
        # n_segments > n_orig_segments
        e_points = 0
        n_unsegmented = 0
        if no_segment_available:
            false_points_idxs = np.argwhere((a_segments != unsegmented))
            e_points = len(false_points_idxs)
            # unsegmented points within a certain segment
            n_unsegmented = len_points - e_points
            n_errornous_points += e_points
        else:
            # save the assignet segment for next iterations
            assignments[orig_segment] = chosen_segment
            # filter the objects with the wrong segments
            unsegmented_and_false_points_idxs = np.argwhere(
                a_segments != chosen_segment)
            # some points can be wrong but unsegmented
            false_points_idxs = np.argwhere(
                a_segments[unsegmented_and_false_points_idxs] != unsegmented)

            # increment the n_errornous_points for every false point
            e_points = len(false_points_idxs)
            # unsegmented points within a certain segment
            n_unsegmented = len(unsegmented_and_false_points_idxs) - e_points
            n_errornous_points += e_points
        # relative number of points that are correct within a segment
        # segment_rewards[i] = 1 - ((n_unsegmented + e_points) / len_points)
        segment_rewards[orig_segment] =\
            (len_points - (n_unsegmented + e_points)) / assigned_segments.shape[0]
        # segment_rewards[i] /= n_orig_segments
        if n_unsegmented == 0:
            finalized_segments.append(i)
    if n_unsegmented_points > 0 and diff > 0:
        diff -= 1
    return\
        n_errornous_points,\
        n_unsegmented_points,\
        diff,\
        assignments,\
        segment_rewards,\
        finalized_segments


def get_interval(orig_seg_idx, orig_indices, orig_segment_counts):
    """Interval of a certain ground truth segment value.

    Parameters
    ----------
    orig_seg_idx : int
        Index of a certain ground truth segment.
    orig_indices : np.ndarray
        Start indices of a sorted ground truth segments array. The indices can
        result from a np.uniquee operation.
    orig_segment_counts : np.ndarray
        Counts of the ground truth segment values. The counts can result from a
        np.uniquee operation.

    Returns
    -------
    tuple(int, int, int)
        Start index, length and stop index of a certain ground truth segment.

    """
    start = orig_indices[orig_seg_idx]
    length = orig_segment_counts[orig_seg_idx]
    stop = start + length
    return start, stop, length


def compute_error1(
        assigned_segments,
        orig_segment_values,
        orig_indices,
        orig_segment_counts,
        assignments,
        unsegmented=0):
    """Computes the number of errorornous and unsegmented points after a
    segmentation of a point cloud. The objects will be evaluated according to
    their size. The largest object will be evaluated first.

    Parameters
    ----------
    assigned_segments : np.ndarray
        Assigned segments vector.
    orig_segment_values : np.ndarray
        The true segments vector. Should be sorted according to segment values.
    orig_indices : np.ndarray
        Description of parameter `orig_indices`.
    orig_segment_counts : np.ndarray
        Description of parameter `orig_segment_counts`.
    unsegmented : int
        Number that indicates an unsegmented point.

    Returns
    -------
    tuple(int, int, int, dict(int, int), np.ndarray, list(int))
        Number of errorornous points. Number of unsegmented points. Difference
        of the number of objects in the true and assigned segments vectors.
        Mapping from a segment of the true segments vector to a segment number
        of the assigned segments vector. Array where a reward is stored for
        each segment. List of segments that could be rewarded.

    """
    intervals = []

    _, assigned_segment_counts = np.unique(
        assigned_segments, return_counts=True)
    n_orig_segments = orig_segment_values.shape[0]
    # how much points are correct within the i-th segment?
    segment_rewards = np.zeros((n_orig_segments, ), np.float32)
    # no more unsegmented points within segment
    finalized_segments = []
    diff = np.absolute(
        n_orig_segments - assigned_segment_counts.shape[0])
    for i in range(n_orig_segments):
        # start idx of a original segment
        idx = orig_indices[i]
        # how many original segments follow after the start idx
        count = orig_segment_counts[i]
        # np array of estimated in the range of a original segment
        obj_segments = assigned_segments[idx:idx + count]
        # which estimated segments exist in the range of a original segment and
        # how often they occur
        sorted_segments, obj_counts = np.unique(
            obj_segments, return_counts=True)

        # ( int, int, np.array, np.array )
        interval = (
            idx,
            count,
            sorted_segments,
            obj_counts,
            orig_segment_values[i])
        # sort the intervals by considering the length of a segment
        insert_idx = 0
        insert = False
        for j in range(len(intervals)):
            insert_idx = j
            # number of points in one segment
            len_points = intervals[j][1]
            if count > len_points:
                insert = True
                break
        if insert:
            intervals.insert(insert_idx, interval)
        else:
            intervals.append(interval)
    """ check sortation
    for i in range(len(intervals)-1):
        len_points = intervals[i][1]
        len_points_1 = intervals[i+1][1]
        if len_points_1 > len_points:
            raise Exception("Sortation Error: {0} > {1}".format(
                len_points_1, len_points))
    """
    n_unsegmented_points = len(np.argwhere(assigned_segments == unsegmented))
    # assignments = {}
    n_errornous_points = 0
    for i in range(len(intervals)):
        # orig
        obj_idx = intervals[i][0]
        len_points = intervals[i][1]
        orig_segment = intervals[i][4]
        # estimated
        a_sorted_segments = intervals[i][2]
        a_counts = intervals[i][3]

        # estimated segments over the range of a true segment
        a_segments = assigned_segments[obj_idx:obj_idx+len_points]

        # flag that indicates if a segment is already assigned
        is_already_assigned = True
        # flag that indicates if a segment is available
        no_segment_available = False

        while is_already_assigned:
            # no more segment available for assignment
            # this happens if there are less cluster than predicted
            if a_counts.size == 0:
                no_segment_available = True
                break
            # get the index of the most frequent cluster
            j = np.argmax(a_counts)
            # get the most frequent cluster segment
            if a_sorted_segments[j] == unsegmented:
                is_already_assigned = True
            else:
                chosen_segment = a_sorted_segments[j]
                # check if segment is already assigned
                is_already_assigned = chosen_segment in assignments.values()
            if(is_already_assigned):
                # if so delete the segment and take the
                # second most segment and so forth
                a_sorted_segments = np.delete(a_sorted_segments, j)
                a_counts = np.delete(a_counts, j)
            else:
                break
        # if there are no more segments,
        # consider all the points as misclustered
        # n_segments > n_orig_segments
        e_points = 0
        n_unsegmented = 0
        if no_segment_available:
            false_points_idxs = np.argwhere(a_segments != unsegmented)
            e_points = len(false_points_idxs)
            # unsegmented points within a certain segment
            n_unsegmented = len_points - e_points
            n_errornous_points += e_points
        else:
            # save the assignet segment for next iterations
            assignments[orig_segment] = chosen_segment
            # filter the objects with the wrong segments
            unsegmented_and_false_points_idxs = np.argwhere(
                a_segments != chosen_segment)
            # some points can be wrong but unsegmented
            false_points_idxs = np.argwhere(
                a_segments[unsegmented_and_false_points_idxs] != unsegmented)

            # increment the n_errornous_points for every false point
            e_points = len(false_points_idxs)
            # unsegmented points within a certain segment
            n_unsegmented = len(unsegmented_and_false_points_idxs) - e_points
            n_errornous_points += e_points
        # relative number of points that are correct within a segment
        segment_rewards[i] = 1 - ((n_unsegmented + e_points) / len_points)
        segment_rewards[i] /= n_orig_segments
        if n_unsegmented == 0:
            finalized_segments.append(i)
    if n_unsegmented_points > 0 and diff > 0:
        diff -= 1
    return\
        n_errornous_points,\
        n_unsegmented_points,\
        diff,\
        assignments,\
        segment_rewards,\
        finalized_segments
